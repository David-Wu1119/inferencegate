"""Qualify a serving change: replay a workload against two endpoints, decide.

This is the layer between the replay harness and the release gate. Given a
baseline endpoint and a candidate endpoint, it replays the same workload
against both several times, reduces each run to performance, quality, cost and
reliability metrics, pairs the runs, and asks the gate for one verdict.

    python3 qualify.py --baseline-url http://host:8000/v1 \
                       --candidate-url http://host:8001/v1 \
                       --model Qwen/Qwen3-8B --requests reqs.jsonl \
                       --repeats 5 --out runs/

Exit status is the verdict, so a CI job can gate a deploy on it:
0 PASS, 1 BLOCK, 2 INCONCLUSIVE. Anything that stops the qualification from
completing exits 3 (ERROR) and writes no verdict, so a crash can never pass
for a decision.

Runs are **interleaved** (baseline, candidate, baseline, candidate, ...) rather
than run as one block each. A machine drifts over a qualification session from
thermal state, page cache and noisy neighbours, and blocking the arms lets that
drift land entirely on one of them, which is indistinguishable from the change
under test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from grade import FALLBACK, STRUCTURED, grade  # noqa: E402
from release_gate import (  # noqa: E402
    BLOCK,
    ERROR,
    INCONCLUSIVE,
    PASS,
    combine,
    decide,
    decide_paired_grid,
)

HERE = Path(__file__).resolve().parent
REPLAYER = HERE / "run_openai_chat_open_loop.py"
EXIT = {PASS: 0, BLOCK: 1, INCONCLUSIVE: 2, ERROR: 3}
RUN_NAME = re.compile(r"^(baseline|candidate)_r(\d+)\.jsonl$")


def fail(message: str) -> SystemExit:
    """The qualification could not be completed: ERROR, exit 3, no verdict.

    Never exit 1 for a fault. A shell reads 1 as BLOCK, and a crash that
    passes for a decision is a policy decision nobody made.
    """
    print(f"error: {message}", file=sys.stderr)
    return SystemExit(EXIT[ERROR])


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def replay(
    url: str, args, run_id: str, out: Path, seed: int, model: str | None = None
) -> Path:
    """One replay of the workload against one endpoint."""
    cmd = [
        sys.executable,
        str(REPLAYER),
        "--base-url",
        url,
        "--model",
        model or args.model,
        "--engine",
        args.engine,
        "--condition",
        args.condition,
        "--run-id",
        run_id,
        "--requests",
        str(args.requests),
        "--out",
        str(out),
        "--arrival-rate",
        str(args.arrival_rate),
        "--seed",
        str(seed),
        "--max-tokens",
        str(args.max_tokens),
        "--stream",
    ]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise fail(
            f"replay {run_id} failed, so the comparison would rest on a "
            f"partial run:\n{done.stderr.strip()[-800:]}"
        )
    return out


def load_workload(
    requests: Path,
) -> tuple[list[str], dict[str, tuple[str, str, bool, str]]]:
    """Every request id in the workload, and the expectation of each graded one.

    Expectations are joined from the workload file rather than read off the
    response record: the replayer does not echo the expectation back, so
    trusting the record silently grades every item as wrong and makes the
    quality metric look identical across arms. The id list is what every run
    is checked against, so a lost record cannot disappear quietly.
    """
    ids: list[str] = []
    expectations: dict[str, tuple[str, str, bool, str]] = {}
    seen: set[str] = set()
    try:
        lines = requests.read_text().splitlines()
    except OSError as exc:
        raise fail(f"cannot read the workload {requests}: {exc}") from None
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise fail(f"{requests}: malformed request line ({exc})") from None
        rid = row.get("id")
        if rid is None:
            raise fail(f"{requests}: a request has no id")
        if rid in seen:
            raise fail(
                f"{requests}: duplicate request id {rid!r}; two identical items "
                f"are one observation counted twice"
            )
        seen.add(rid)
        ids.append(rid)
        spec = row.get("expected") or {}
        expected = spec.get("answer")
        if expected is not None:
            expectations[rid] = (
                str(expected),
                spec.get("answer_path") or "answer",
                bool(spec.get("json_required_keys")),
                row.get("task_family") or "all",
            )
    return ids, expectations


def load_records(path: Path, run_id: str) -> list[dict]:
    """One run's records: each request exactly once, all from the named run."""
    try:
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    except (OSError, ValueError) as exc:
        raise fail(f"{path.name}: unreadable run records ({exc})") from None
    seen: set[str] = set()
    for r in records:
        rid = r.get("id")
        if rid is None:
            raise fail(f"{path.name}: a record has no request id")
        if rid in seen:
            raise fail(
                f"{path.name}: duplicate request id {rid!r}; a run must hold "
                f"each request exactly once, so this run cannot be paired"
            )
        seen.add(rid)
        if r.get("run_id") != run_id:
            raise fail(
                f"{path.name}: record {rid!r} carries run_id {r.get('run_id')!r}, "
                f"expected {run_id!r}; the file name and its records disagree "
                f"about which run this is"
            )
    return records


def check_workload(records: list[dict], workload_ids: list[str], name: str) -> None:
    """A run must answer every workload request, and nothing else."""
    have = {r["id"] for r in records}
    missing = sorted(set(workload_ids) - have)
    extra = sorted(have - set(workload_ids))

    def some(ids: list[str]) -> str:
        return ", ".join(ids[:5]) + (", ..." if len(ids) > 5 else "")

    if missing:
        raise fail(
            f"{name}: missing {len(missing)} of {len(workload_ids)} workload "
            f"requests ({some(missing)}); a lost record is not a failed "
            f"request, and the run cannot be paired"
        )
    if extra:
        raise fail(
            f"{name}: {len(extra)} records for requests not in the workload "
            f"({some(extra)}); the run and the workload do not match"
        )


def saved_runs(out: Path) -> list[tuple[int, Path, Path]]:
    """The saved repeats in ``out``, paired by repeat number, never by order."""
    found: dict[int, dict[str, Path]] = {}
    for path in out.glob("*.jsonl"):
        m = RUN_NAME.match(path.name)
        if m:
            found.setdefault(int(m.group(2)), {})[m.group(1)] = path
    if not found:
        raise fail(
            f"no saved runs named baseline_rK.jsonl / candidate_rK.jsonl in {out}"
        )
    unpaired = sorted(k for k, arms in found.items() if len(arms) != 2)
    if unpaired:
        raise fail(
            f"{out}: repeat(s) {unpaired} have one arm only; every baseline_rK "
            f"needs its candidate_rK"
        )
    return [(k, found[k]["baseline"], found[k]["candidate"]) for k in sorted(found)]


def reduce_run(
    path: Path,
    run_id: str,
    workload_ids: list[str],
    expectations: dict[str, tuple[str, str, bool, str]],
    expect: str | None = None,
    slo_ttft_ms: float = 500.0,
    slo_tpot_ms: float = 30.0,
) -> dict:
    """Turn one run's per-request records into run-level metrics.

    Per-request outcomes keep their request id so they can be paired by
    identity downstream. A request that failed stays in as a failed
    observation: not successful, not correct, not compliant.
    """
    records = load_records(path, run_id)
    check_workload(records, workload_ids, path.name)
    ttft = [
        r["latency"]["ttft_ms"] for r in records if r.get("latency", {}).get("ttft_ms")
    ]
    tpot = [
        r["latency"]["tpot_ms"] for r in records if r.get("latency", {}).get("tpot_ms")
    ]
    errors = sum(1 for r in records if r.get("error"))
    starts = [r["request_started_at_unix"] for r in records]
    ends = [r["request_completed_at_unix"] for r in records]
    makespan = max(ends) - min(starts) if records else float("nan")
    out_tokens = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in records)

    good = 0
    for r in records:
        lat = r.get("latency") or {}
        if (
            lat.get("ttft_ms") is not None
            and lat["ttft_ms"] <= slo_ttft_ms
            and lat.get("tpot_ms") is not None
            and lat["tpot_ms"] <= slo_tpot_ms
        ):
            good += 1

    quality = {}
    compliance = {}
    families = {}
    # A server error is a reliability regression to be gated, not a reason
    # for the qualification itself to fall over.
    success = {r["id"]: not r.get("error") for r in records}
    how_counts = {STRUCTURED: 0, FALLBACK: 0}
    for r in records:
        spec = expectations.get(r["id"])
        if spec is None:
            if expect is None:
                continue
            expected, answer_path, declared, family = expect, "answer", False, "all"
        else:
            expected, answer_path, declared, family = spec
        families[r["id"]] = family
        correct, how = grade(r.get("output_text") or "", expected, answer_path)
        quality[r["id"]] = correct
        # Returning the declared field is its own outcome: a change can keep
        # answers right and stop honouring the response format, which needs a
        # different fix than a knowledge regression. Only counted where the
        # workload actually asked for a format, since a model that volunteers
        # JSON nobody requested is not violating a contract by dropping it.
        if declared:
            compliance[r["id"]] = how == STRUCTURED
        if how in how_counts:
            how_counts[how] += 1

    return {
        "run_id": run_id,
        "n": len(records),
        "errors": errors,
        "graded_structured": how_counts[STRUCTURED],
        "graded_fallback": how_counts[FALLBACK],
        "compliance": compliance,
        "success": success,
        "families": families,
        "throughput_rps": (
            sum(success.values()) / makespan if makespan > 0 else float("nan")
        ),
        "goodput_rps": good / makespan if makespan > 0 else float("nan"),
        "p95_ttft_ms": percentile(ttft, 0.95),
        "p95_tpot_ms": percentile(tpot, 0.95),
        "sec_per_1k_output_tokens": (makespan / (out_tokens / 1000.0))
        if out_tokens
        else float("nan"),
        "quality": quality,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str | None:
    """The checkout's HEAD, if this is a git checkout; the file hashes beside
    it say whether the gate files match that revision."""
    try:
        done = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def build_grids(
    base_runs: list[dict], cand_runs: list[dict]
) -> tuple[dict[str, dict[str, list[tuple[bool, bool]]]], dict[str, dict]]:
    """Pair per-request outcomes by prompt id within each session.

    Returns ``grids[metric][prompt_id] = [(baseline, candidate), ...]`` in
    session order, plus the quality grid split by task family. Both runs of a
    session were checked against the same workload, so pairing is by
    identity and a prompt present in one arm but not the other is a fault.
    """
    grids: dict[str, dict[str, list[tuple[bool, bool]]]] = {
        "quality": {},
        "compliance": {},
        "success": {},
    }
    families: dict[str, str] = {}
    for b, c in zip(base_runs, cand_runs, strict=True):
        for metric, grid in grids.items():
            for pid, outcome in b[metric].items():
                if pid not in c[metric]:
                    raise fail(
                        f"{pid!r} has a {metric} outcome in {b['run_id']} but "
                        f"not in {c['run_id']}"
                    )
                grid.setdefault(pid, []).append((outcome, c[metric][pid]))
        families.update(b["families"])
    by_family: dict[str, dict] = {}
    for pid, pairs in grids["quality"].items():
        by_family.setdefault(families.get(pid, "all"), {})[pid] = pairs
    return grids, by_family


def main() -> int:
    ap = argparse.ArgumentParser()

    def usage_error(message: str) -> None:
        # argparse exits 2 on a bad flag, which a shell would read as
        # INCONCLUSIVE. A mistyped invocation measured nothing: ERROR.
        ap.print_usage(sys.stderr)
        raise fail(message)

    ap.error = usage_error  # type: ignore[method-assign]
    ap.add_argument("--baseline-url", default=None)
    ap.add_argument("--candidate-url", default=None)
    ap.add_argument(
        "--model",
        required=True,
        help="Model id sent to both arms unless overridden below.",
    )
    ap.add_argument(
        "--baseline-model",
        default=None,
        help="Model id for the baseline arm. A model or "
        "quantization change makes the two arms differ, and "
        "an endpoint that resolves the id per request will "
        "otherwise serve the wrong weights.",
    )
    ap.add_argument("--candidate-model", default=None)
    ap.add_argument("--requests", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--engine", default="vllm", choices=["vllm", "sglang"])
    ap.add_argument("--condition", default="prompt_json")
    ap.add_argument("--arrival-rate", type=float, default=20.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--slo-ttft-ms", type=float, default=500.0)
    ap.add_argument("--slo-tpot-ms", type=float, default=30.0)
    ap.add_argument(
        "--expect",
        default=None,
        help="Substring a correct answer must contain, when the "
        "workload carries no per-item expectation.",
    )
    # Pre-registered margins: the smallest regression worth blocking a deploy.
    ap.add_argument("--margin-goodput-rps", type=float, default=1.0)
    ap.add_argument("--margin-throughput-rps", type=float, default=1.0)
    ap.add_argument("--margin-p95-ttft-ms", type=float, default=50.0)
    ap.add_argument("--margin-p95-tpot-ms", type=float, default=5.0)
    ap.add_argument(
        "--margin-cost",
        type=float,
        default=0.5,
        help="smallest rise in wall-clock seconds per 1k output tokens "
        "worth blocking a deploy. A time proxy for cost on fixed "
        "hardware, not GPU utilization and not dollars.",
    )
    ap.add_argument("--margin-quality", type=float, default=0.02)
    ap.add_argument(
        "--margin-success",
        type=float,
        default=0.01,
        help="smallest drop in the fraction of requests that "
        "succeed that is worth blocking a deploy",
    )
    ap.add_argument(
        "--margin-compliance",
        type=float,
        default=0.05,
        help="smallest drop in responses honouring the declared "
        "answer format that is worth blocking a deploy",
    )
    ap.add_argument(
        "--session-failure-margin",
        type=float,
        default=None,
        help="largest per-session probability of an intermittent "
        "whole-session regression the verdict must exclude. Sessions "
        "that all agree cannot exclude one: 5 sessions leave 45%% "
        "open, and no number of prompts lowers that. Declared, a PASS "
        "on quality, format or success needs enough sessions "
        "(29 with none failing for 0.10 on format and success; the "
        "mean-quality claim also charges the unobserved failure tail, "
        "so it needs more) and is INCONCLUSIVE until then. Required "
        "for a release verdict: without it, or --diagnostic, the gate "
        "exits 3.",
    )
    ap.add_argument(
        "--diagnostic",
        action="store_true",
        help="observed-sessions analysis that reports but cannot approve: "
        "every prompt-population decision that would PASS is reported "
        "INCONCLUSIVE with its diagnostic result, and the verdict carries "
        "scope=observed-sessions-diagnostic. A release verdict needs "
        "--session-failure-margin instead.",
    )
    ap.add_argument(
        "--from-existing",
        action="store_true",
        help="Skip replay and re-gate the runs already in --out, "
        "so margins can be revised without new GPU time.",
    )
    args = ap.parse_args()
    # A release verdict needs a declared session-risk policy: sessions that
    # all agree cannot exclude an intermittent whole-session regression, and
    # a verdict that never said what it excludes is not a release decision.
    # Missing configuration is ERROR; insufficient evidence is INCONCLUSIVE.
    if args.diagnostic and args.session_failure_margin is not None:
        raise fail("--diagnostic and --session-failure-margin are exclusive")
    if not args.diagnostic and args.session_failure_margin is None:
        raise fail(
            "a release verdict needs a declared session-failure margin: pass "
            "--session-failure-margin S (for example 0.10, the largest "
            "per-session probability of a whole-session regression the "
            "verdict must exclude), or --diagnostic for an observed-sessions "
            "analysis that reports but cannot approve"
        )
    if not args.from_existing and not (args.baseline_url and args.candidate_url):
        raise fail(
            "--baseline-url and --candidate-url are required "
            "unless --from-existing re-gates saved runs"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    # A re-gate must never overwrite the verdict the run itself produced: that
    # file is evidence, and a check that rewrites what it checks destroys the
    # thing it was meant to confirm. Re-gated verdicts go beside it. Either
    # way, whatever an earlier run left at the target path is removed before
    # anything can fail, so a crash leaves no verdict to mistake for a result.
    verdict_name = "verdict.regated.json" if args.from_existing else "verdict.json"
    (args.out / verdict_name).unlink(missing_ok=True)
    workload_ids, expectations = load_workload(args.requests)

    def reduce(path: Path, run_id: str) -> dict:
        return reduce_run(
            path,
            run_id,
            workload_ids,
            expectations,
            args.expect,
            args.slo_ttft_ms,
            args.slo_tpot_ms,
        )

    base_runs, cand_runs = [], []
    run_paths: list[Path] = []
    if args.from_existing:
        # Re-gate runs already on disk. Margins are a policy choice an operator
        # revises after seeing a verdict, and re-renting a GPU to change one
        # number is the wrong price for that.
        for k, b_path, c_path in saved_runs(args.out):
            base_runs.append(reduce(b_path, f"baseline_r{k}"))
            cand_runs.append(reduce(c_path, f"candidate_r{k}"))
            run_paths += [b_path, c_path]
        args.repeats = len(base_runs)
        print(f"re-gating {len(base_runs)} saved repeats per arm from {args.out}")
    else:
        for r in range(1, args.repeats + 1):
            # Interleaved so session drift hits both arms equally.
            for side, url, bucket, model in (
                ("baseline", args.baseline_url, base_runs, args.baseline_model),
                ("candidate", args.candidate_url, cand_runs, args.candidate_model),
            ):
                path = replay(
                    url,
                    args,
                    f"{side}_r{r}",
                    args.out / f"{side}_r{r}.jsonl",
                    seed=r,
                    model=model,
                )
                bucket.append(reduce(path, f"{side}_r{r}"))
                run_paths.append(path)
            print(f"  repeat {r}/{args.repeats} done", file=sys.stderr)

    metrics = [
        ("goodput_rps", args.margin_goodput_rps, True),
        ("throughput_rps", args.margin_throughput_rps, True),
        ("p95_ttft_ms", args.margin_p95_ttft_ms, False),
        ("p95_tpot_ms", args.margin_p95_tpot_ms, False),
        ("sec_per_1k_output_tokens", args.margin_cost, False),
    ]
    decisions = []
    for name, margin, higher_better in metrics:
        deltas = [c[name] - b[name] for b, c in zip(base_runs, cand_runs, strict=True)]
        decisions.append(
            decide(deltas, margin, metric=name, higher_is_better=higher_better)
        )

    # Quality, success and format are per-prompt outcomes paired by prompt id
    # within each session. The claim is about the population of prompts the
    # workload stands for, so the estimator counts prompts as its units and
    # treats a session as another look at the same prompts, with its own
    # variance component. See docs/gate_repair_2026-09/METHOD_NOTE.md.
    grids, by_family = build_grids(base_runs, cand_runs)
    if grids["quality"]:
        decisions.append(
            decide_paired_grid(
                grids["quality"],
                args.margin_quality,
                metric="answer_quality",
                session_failure_margin=args.session_failure_margin,
                diagnostic=args.diagnostic,
                mean_quality_claim=True,
            )
        )
    if grids["success"]:
        decisions.append(
            decide_paired_grid(
                grids["success"],
                args.margin_success,
                metric="request_success",
                session_failure_margin=args.session_failure_margin,
                diagnostic=args.diagnostic,
            )
        )

    # A single averaged quality number is sensitive to which task families the
    # workload happens to contain: a change that wrecks arithmetic while
    # improving word ordering averages to nearly nothing, and adding easy
    # families dilutes any regression toward the margin. So each family is
    # gated too, and a release that destroys one capability is blocked even
    # when the mean survives. The confidence level is split across the
    # families tested, or the chance that one of a dozen blocks a healthy
    # release would grow with the size of the eval set.
    family_decisions = []
    family_alpha = 0.05 / len(by_family) if len(by_family) > 1 else 0.05
    if len(by_family) > 1:
        for fam, grid in sorted(by_family.items()):
            family_decisions.append(
                decide_paired_grid(
                    grid,
                    args.margin_quality,
                    metric=f"quality[{fam}]",
                    alpha=family_alpha,
                    session_failure_margin=args.session_failure_margin,
                    diagnostic=args.diagnostic,
                    mean_quality_claim=True,
                )
            )

    if grids["compliance"]:
        decisions.append(
            decide_paired_grid(
                grids["compliance"],
                args.margin_compliance,
                metric="schema_compliance",
                session_failure_margin=args.session_failure_margin,
                diagnostic=args.diagnostic,
            )
        )
    elif grids["quality"]:
        print(
            "note: no workload item declares json_required_keys, so response "
            "format is not gated; see the per-arm grading shares below"
        )

    # Reliability: any error at all in either arm is worth surfacing.
    err_b = sum(r["errors"] for r in base_runs)
    err_c = sum(r["errors"] for r in cand_runs)

    # One contract for every layer: any BLOCK blocks, otherwise any
    # INCONCLUSIVE holds, and only an all-PASS set passes. Families count
    # exactly like metrics. A family nobody could resolve is a capability
    # nobody verified, and the 2026-09-05 audit found a release passing on
    # precisely that: the held family was printed, then ignored.
    gated = decisions + family_decisions
    verdict = combine(d.verdict for d in gated)
    blocked_by = [d.metric for d in gated if d.verdict == BLOCK]
    held_by = [d.metric for d in gated if d.verdict == INCONCLUSIVE]

    print("\n" + "=" * 78)
    print(f"Release qualification: {args.repeats} interleaved repeats per arm")
    print("=" * 78)
    for d in decisions:
        print(d)
    if family_decisions:
        blocked = [d for d in family_decisions if d.verdict == BLOCK]
        print(
            f"\nper-family quality, {len(by_family)} families at "
            f"alpha={0.05 / len(by_family):.4f} (Bonferroni):"
        )
        for d in sorted(family_decisions, key=lambda x: x.mean_delta):
            print(
                f"  {d.verdict:13} {d.metric:26} {d.unique_prompts:>4} prompts "
                f"x {d.repeats} delta={d.mean_delta:+.3f} "
                f"CI{100 * (1 - d.alpha):.4g}%=[{d.ci_lo:+.3f},{d.ci_hi:+.3f}]"
            )
        if blocked:
            print(
                f"  {len(blocked)} family/families block even though the "
                f"averaged quality metric may not"
            )
        held = [d for d in family_decisions if d.verdict == INCONCLUSIVE]
        if held:
            print(
                f"  {len(held)} family/families could not be resolved and "
                f"hold the release: " + ", ".join(d.metric for d in held)
            )

    print("-" * 78)
    print(f"errors: baseline {err_b}, candidate {err_c}")

    # Grading provenance: a quality number reached mostly by text search is
    # not a quality number worth gating on.
    shares = {}
    for arm, runs in (("baseline", base_runs), ("candidate", cand_runs)):
        st = sum(r["graded_structured"] for r in runs)
        fb = sum(r["graded_fallback"] for r in runs)
        if st + fb:
            shares[arm] = 100.0 * st / (st + fb)
            print(
                f"grading {arm:9}: {st}/{st + fb} items graded from the "
                f"declared JSON answer field ({shares[arm]:.1f}%), "
                f"{fb} by text search"
            )
    structured = sum(r["graded_structured"] for r in base_runs + cand_runs)
    fallback = sum(r["graded_fallback"] for r in base_runs + cand_runs)
    # Reported per arm rather than pooled: an arm that stops honouring the
    # format drags its own quality number onto the text-search path, and a
    # pooled average is exactly where that disappears.
    for arm, share in shares.items():
        if share < 80.0:
            hint = (
                "see the schema_compliance verdict above"
                if grids["compliance"]
                else "declare json_required_keys in the workload to gate "
                "this instead of only warning about it"
            )
            print(
                f"  WARNING: the {arm} arm returned the declared field for "
                f"only {share:.1f}% of items, so its quality number leans "
                f"on text search; {hint}"
            )
    print(f"\nVERDICT: {verdict}")
    if blocked_by:
        print("  blocked by: " + ", ".join(blocked_by))
    if held_by:
        print("  held by: " + ", ".join(held_by))
    risk = [d for d in gated if d.session_rate_upper is not None]
    if risk:
        d = max(risk, key=lambda x: x.session_rate_upper)
        if args.diagnostic:
            print(
                f"  scope: observed-sessions diagnostic over {d.repeats} session(s); "
                f"cannot approve a release"
            )
        else:
            print(
                f"  session policy: failure margin {args.session_failure_margin:g}; "
                f"worst bound on the per-session failure rate "
                f"{d.session_rate_upper:.3f} from {d.session_failures_observed} "
                f"failure(s) in {d.repeats} session(s)"
            )
    if args.from_existing:
        print(
            f"  written to {args.out / verdict_name} (the run's own "
            f"verdict.json is left untouched)"
        )
    if verdict == INCONCLUSIVE:
        for d in gated:
            if d.verdict != INCONCLUSIVE:
                continue
            if d.runs_needed:
                print(
                    f"  {d.metric}: about {d.runs_needed} repeats per arm "
                    f"would settle it"
                )
            else:
                print(f"  {d.metric}: {d.reason}")
    (args.out / verdict_name).write_text(
        json.dumps(
            {
                "verdict": verdict,
                "scope": "observed-sessions-diagnostic"
                if args.diagnostic
                else "release",
                "blocked_by": blocked_by,
                "held_by": held_by,
                "sampling": {
                    "workload_requests": len(workload_ids),
                    "graded_prompts": len(grids["quality"]),
                    "families": len(by_family),
                    "repeats": args.repeats,
                    "observations_per_arm": sum(r["n"] for r in base_runs),
                },
                "decisions": [d.__dict__ for d in decisions],
                "family_decisions": [d.__dict__ for d in family_decisions],
                "errors": {"baseline": err_b, "candidate": err_c},
                "grading": {
                    "structured": structured,
                    "fallback": fallback,
                    "structured_share_by_arm": shares,
                },
                # What this verdict is a verdict about: the exact records and
                # workload it read, the policy it applied, and the code that
                # applied it. A report that cannot say this cannot be checked.
                "inputs": {
                    "requests": {
                        "path": str(args.requests),
                        "sha256": sha256(args.requests),
                    },
                    "runs": {p.name: sha256(p) for p in sorted(run_paths)},
                },
                "policy": {
                    "margins": {
                        "goodput_rps": args.margin_goodput_rps,
                        "throughput_rps": args.margin_throughput_rps,
                        "p95_ttft_ms": args.margin_p95_ttft_ms,
                        "p95_tpot_ms": args.margin_p95_tpot_ms,
                        "cost": args.margin_cost,
                        "quality": args.margin_quality,
                        "success": args.margin_success,
                        "compliance": args.margin_compliance,
                    },
                    "slo": {"ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms},
                    "alpha": 0.05,
                    "family_alpha": family_alpha,
                    "families": len(by_family),
                    "session_failure_margin": args.session_failure_margin,
                },
                "implementation": {
                    "git_revision": git_revision(),
                    "release_gate.py": sha256(
                        HERE.parent / "analysis" / "release_gate.py"
                    ),
                    "qualify.py": sha256(Path(__file__).resolve()),
                },
            },
            indent=2,
        )
    )
    return EXIT[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
