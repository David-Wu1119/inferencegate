"""Verdict propagation from saved records, with no server.

Reproduces the false PASS from the 2026-09-05 audit. The fixture is 599
prompts in one task family and one prompt in another, five repeats per arm,
both arms correct, schema-compliant, error-free and equally fast. Every
aggregate metric passes on those numbers. The lone-prompt family cannot be
resolved at any margin, and a release with an unresolved family is
INCONCLUSIVE, exit 2 -- not PASS, exit 0.

The common family is large enough that the aggregate quality metric stays PASS
when quality is estimated over prompts rather than over pooled observations, so
this test cannot start passing because everything went INCONCLUSIVE. The
prerequisite assertions guard that: if the fixture stops producing an all-PASS
aggregate, the failure names the fixture, not the gate.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUALIFY = HERE / "qualify.py"

AGGREGATE_METRICS = (
    "goodput_rps",
    "throughput_rps",
    "p95_ttft_ms",
    "p95_tpot_ms",
    "sec_per_1k_output_tokens",
    "answer_quality",
    "request_success",
    "schema_compliance",
)


def write_fixture(
    work: Path, repeats: int = 5, common: int = 599, rare: int = 1
) -> list[dict]:
    """Saved records for two arms that agree on everything.

    Latencies carry small independent jitter per arm and per repeat, so the
    performance intervals are real t-intervals with nonzero width that clear
    their margins, not zero-width intervals from identical numbers.
    """
    rows = [
        {
            "id": f"c{i:03d}",
            "prompt": f"question {i}",
            "expected": {
                "answer": str(i),
                "answer_path": "answer",
                "json_required_keys": ["answer"],
            },
            "task_family": "common",
        }
        for i in range(common)
    ]
    rows += [
        {
            "id": f"r{i:03d}",
            "prompt": f"rare question {i}",
            "expected": {
                "answer": "rare",
                "answer_path": "answer",
                "json_required_keys": ["answer"],
            },
            "task_family": "rare",
        }
        for i in range(rare)
    ]
    (work / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    rng = random.Random(20260905)
    for r in range(1, repeats + 1):
        for arm in ("baseline", "candidate"):
            t0 = 1_800_000_000.0
            lines = []
            for k, row in enumerate(rows):
                ttft = rng.gauss(100.0, 3.0)
                tpot = rng.gauss(6.0, 0.3)
                tokens = 9
                start = t0 + 0.05 * k
                lines.append(
                    json.dumps(
                        {
                            "id": row["id"],
                            "run_id": f"{arm}_r{r}",
                            "latency": {"ttft_ms": ttft, "tpot_ms": tpot},
                            "usage": {"input_tokens": 20, "output_tokens": tokens},
                            "output_text": json.dumps(
                                {"answer": row["expected"]["answer"]}
                            ),
                            "request_started_at_unix": start,
                            "request_completed_at_unix": start
                            + (ttft + tpot * tokens) / 1000.0,
                        }
                    )
                )
            (work / f"{arm}_r{r}.jsonl").write_text("\n".join(lines) + "\n")
    return rows


def regate(
    work: Path,
    expect_report: bool = True,
    diagnostic: bool = False,
    policy: float | None = 0.10,
    **margins: float,
):
    """Re-gate saved records. A release verdict needs a session policy, so
    one is declared unless the call is a diagnostic or passes policy=None to
    exercise the missing-policy error."""
    cmd = [
        sys.executable,
        str(QUALIFY),
        "--from-existing",
        "--model",
        "fixture",
        "--requests",
        str(work / "requests.jsonl"),
        "--out",
        str(work),
    ]
    if "session_failure" in margins:
        policy = margins.pop("session_failure")
    for name, value in margins.items():
        cmd += [f"--margin-{name.replace('_', '-')}", str(value)]
    if diagnostic:
        cmd.append("--diagnostic")
    elif policy is not None:
        cmd += ["--session-failure-margin", str(policy)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    report = work / "verdict.regated.json"
    if not expect_report:
        return (json.loads(report.read_text()) if report.exists() else None), done
    assert report.exists(), f"no verdict written; stderr:\n{done.stderr[-1200:]}"
    return json.loads(report.read_text()), done


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_records(path: Path, recs: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def assert_rejected(v, done, *needles: str) -> None:
    """A pairing fault must end the qualification, not reach a verdict."""
    print(f"  exit {done.returncode}, verdict written: {v is not None}")
    assert v is None, v
    assert done.returncode not in (0, 1, 2), (done.returncode, done.stderr[-600:])
    for needle in needles:
        assert needle in done.stderr, (needle, done.stderr[-800:])


def test_duplicate_id_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_fixture(work, common=39)
        path = work / "candidate_r2.jsonl"
        recs = records(path)
        write_records(path, recs + [recs[0]])
        v, done = regate(work, expect_report=False, quality=0.10)
        assert_rejected(v, done, "duplicate", "c000", "candidate_r2")


def test_missing_record_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_fixture(work, common=39)
        path = work / "baseline_r3.jsonl"
        write_records(path, [r for r in records(path) if r["id"] != "c007"])
        v, done = regate(work, expect_report=False, quality=0.10)
        assert_rejected(v, done, "missing", "c007", "baseline_r3")


def test_mismatched_run_id_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_fixture(work, common=39)
        path = work / "candidate_r4.jsonl"
        write_records(path, [{**r, "run_id": "candidate_r1"} for r in records(path)])
        v, done = regate(work, expect_report=False, quality=0.10)
        assert_rejected(v, done, "run_id", "candidate_r4", "candidate_r1")


def test_failed_requests_are_observations() -> None:
    """A request that failed is a failed observation, counted and gated."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_fixture(work)
        failing = {f"c{i:03d}" for i in range(30)}
        for r in range(1, 6):
            path = work / f"candidate_r{r}.jsonl"
            recs = records(path)
            for rec in recs:
                if rec["id"] in failing:
                    rec.update(
                        error="HTTPError: 503 mock overload",
                        output_text="",
                        latency={},
                        usage={"input_tokens": 20, "output_tokens": 0},
                    )
            write_records(path, recs)
        v, done = regate(work, quality=0.10, compliance=0.05, success=0.01)
    by_metric = {d["metric"]: d for d in v["decisions"]}
    succ = by_metric["request_success"]
    print(
        f"  release {v['verdict']}, exit {done.returncode}; request_success "
        f"{succ['verdict']} delta {succ['mean_delta']:+.3f} over "
        f"{succ['observations']} observations, candidate errors "
        f"{v['errors']['candidate']}"
    )
    assert v["errors"]["candidate"] == 150, v["errors"]
    assert succ["observations"] == 3000 and succ["unique_prompts"] == 600, succ
    assert abs(succ["mean_delta"] + 0.05) < 1e-9, succ
    assert succ["verdict"] == "BLOCK", succ
    assert "request_success" in v["blocked_by"], v["blocked_by"]
    assert (v["verdict"], done.returncode) == ("BLOCK", 1), (v["verdict"], done)
    # The failed requests are wrong and non-compliant too, and counted as such.
    assert by_metric["answer_quality"]["mean_delta"] < 0, by_metric["answer_quality"]
    assert by_metric["schema_compliance"]["mean_delta"] < 0, by_metric


def test_copies_do_not_narrow() -> None:
    """Five sessions of identical records are one session's evidence."""
    with tempfile.TemporaryDirectory() as tmp:
        once, five = Path(tmp) / "once", Path(tmp) / "five"
        once.mkdir()
        five.mkdir()
        write_fixture(once, repeats=1, common=20, rare=0)
        write_fixture(five, repeats=1, common=20, rare=0)
        for arm in ("baseline", "candidate"):
            recs = records(five / f"{arm}_r1.jsonl")
            for r in range(2, 6):
                write_records(
                    five / f"{arm}_r{r}.jsonl",
                    [{**rec, "run_id": f"{arm}_r{r}"} for rec in recs],
                )
        a, _ = regate(once, quality=0.10)
        b, _ = regate(five, quality=0.10)
    qa = {d["metric"]: d for d in a["decisions"]}["answer_quality"]
    qb = {d["metric"]: d for d in b["decisions"]}["answer_quality"]
    print(
        f"  once: {qa['verdict']} over {qa['unique_prompts']} prompts x "
        f"{qa['repeats']}; copied x5: {qb['verdict']} over "
        f"{qb['unique_prompts']} prompts x {qb['repeats']}"
    )
    assert (qa["unique_prompts"], qa["repeats"]) == (20, 1), qa
    assert (qb["unique_prompts"], qb["repeats"]) == (20, 5), qb
    assert (qa["ci_lo"], qa["ci_hi"]) == (qb["ci_lo"], qb["ci_hi"]), (qa, qb)
    assert qa["verdict"] == qb["verdict"] == "INCONCLUSIVE", (qa, qb)


def test_unresolved_family_holds_release() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # 42 sessions: with two families the common family is gated at
        # alpha 0.025, and the session-mean lower bound (round-4 fix) needs
        # about 40 agreeing sessions to clear the 0.10 quality margin once the
        # unobserved-failure tail is charged at -1. 42 gives headroom.
        write_fixture(work, repeats=42)
        v, done = regate(work, quality=0.10, compliance=0.05, success=0.01)

    # Prerequisites: the fixture must produce exactly the audited shape.
    by_metric = {d["metric"]: d for d in v["decisions"]}
    for metric in AGGREGATE_METRICS:
        d = by_metric.get(metric)
        assert d is not None, f"fixture prerequisite: {metric} was not gated"
        assert d["verdict"] == "PASS", f"fixture prerequisite: {metric} is {d}"
    by_family = {d["metric"]: d for d in v["family_decisions"]}
    assert by_family["quality[common]"]["verdict"] == "PASS", by_family
    assert by_family["quality[rare]"]["verdict"] == "INCONCLUSIVE", by_family

    # The regression: a family nobody could resolve must hold the release.
    print(
        f"  release {v['verdict']}, exit {done.returncode}; "
        f"quality[rare] {by_family['quality[rare]']['verdict']}"
    )
    assert v["verdict"] == "INCONCLUSIVE", (
        f"release reported {v['verdict']} with quality[rare] INCONCLUSIVE"
    )
    assert done.returncode == 2, (done.returncode, done.stderr[-600:])
    assert "quality[rare]" in v["held_by"], v.get("held_by")
    assert v["blocked_by"] == [], v.get("blocked_by")


def test_stale_verdict_cannot_survive_crash() -> None:
    """An earlier PASS on disk must not be read as the result of a crashed run."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_fixture(work, common=39)
        (work / "verdict.regated.json").write_text(json.dumps({"verdict": "PASS"}))
        path = work / "candidate_r2.jsonl"
        recs = records(path)
        write_records(path, recs + [recs[0]])
        v, done = regate(work, expect_report=False, quality=0.10)
        assert_rejected(v, done, "duplicate")


def test_report_is_bound_to_inputs() -> None:
    """The verdict names the exact inputs, policy and code that produced it."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_fixture(work, common=39)
        v, _ = regate(work, quality=0.10, compliance=0.05, success=0.01)
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
        assert v["inputs"]["requests"]["sha256"] == digest(work / "requests.jsonl")
        assert v["inputs"]["requests"]["path"].endswith("requests.jsonl")
        runs = v["inputs"]["runs"]
        assert set(runs) == {
            f"{a}_r{k}.jsonl" for a in ("baseline", "candidate") for k in range(1, 6)
        }
        assert runs["candidate_r3.jsonl"] == digest(work / "candidate_r3.jsonl")
    assert v["policy"]["margins"]["quality"] == 0.10, v["policy"]
    assert v["policy"]["margins"]["compliance"] == 0.05, v["policy"]
    assert v["policy"]["family_alpha"] == 0.05 / 2, v["policy"]
    impl = v["implementation"]
    assert impl["release_gate.py"] == digest(
        HERE.parent / "analysis" / "release_gate.py"
    )
    assert impl["qualify.py"] == digest(HERE / "qualify.py")
    assert impl["git_revision"] is None or len(impl["git_revision"]) >= 7, impl
    print(
        f"  bound to {len(runs)} run files, requests {v['inputs']['requests']['sha256'][:12]}, revision {impl['git_revision']}"
    )


def test_nonfinite_measurement_is_error() -> None:
    """A session with no latency at all cannot yield a latency verdict."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_fixture(work, common=39)
        path = work / "candidate_r2.jsonl"
        recs = records(path)
        for rec in recs:
            rec.update(
                error="HTTPError: 503 mock overload",
                output_text="",
                latency={},
                usage={"input_tokens": 20, "output_tokens": 0},
            )
        write_records(path, recs)
        v, done = regate(work, quality=0.10)
    by_metric = {d["metric"]: d for d in v["decisions"]}
    errored = [m for m, d in by_metric.items() if d["verdict"] == "ERROR"]
    print(f"  release {v['verdict']}, exit {done.returncode}; ERROR on {errored}")
    assert done.returncode == 3, (done.returncode, done.stderr[-400:])
    assert v["verdict"] == "ERROR", v["verdict"]
    assert "p95_ttft_ms" in errored and "sec_per_1k_output_tokens" in errored, errored
    assert "not finite" in by_metric["p95_ttft_ms"]["reason"], by_metric["p95_ttft_ms"]
    # The reliability regression is still visible beside the error.
    assert by_metric["request_success"]["mean_delta"] < 0, by_metric["request_success"]


def test_release_requires_a_declared_policy() -> None:
    """No policy: ERROR. Diagnostic: analysis, never approval. Policy: gated."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_fixture(work, common=600, rare=0)
        v, done = regate(work, expect_report=False, policy=None, quality=0.10)
        assert_rejected(v, done, "session-failure margin", "--diagnostic")

        v, done = regate(
            work, quality=0.10, compliance=0.05, success=0.01, diagnostic=True
        )
        aq = {d["metric"]: d for d in v["decisions"]}["answer_quality"]
        print(
            f"  diagnostic: {v['verdict']} exit {done.returncode}, scope "
            f"{v['scope']}; answer_quality {aq['verdict']}"
        )
        assert v["scope"] == "observed-sessions-diagnostic", v["scope"]
        assert (v["verdict"], done.returncode) == ("INCONCLUSIVE", 2), v["verdict"]
        assert aq["verdict"] == "INCONCLUSIVE" and "diagnostic" in aq["reason"], aq
        assert v["policy"]["session_failure_margin"] is None, v["policy"]

        v, done = regate(
            work, quality=0.10, compliance=0.05, success=0.01, session_failure=0.10
        )
        aq = {d["metric"]: d for d in v["decisions"]}["answer_quality"]
        print(
            f"  policy 0.10 on 5 sessions: {v['verdict']} exit {done.returncode}; "
            f"sessions needed {aq['sessions_needed']}"
        )
        assert v["scope"] == "release", v["scope"]
        assert (v["verdict"], done.returncode) == ("INCONCLUSIVE", 2), v["verdict"]
        assert aq["sessions_needed"] == 29, aq
        assert v["policy"]["session_failure_margin"] == 0.10, v["policy"]


def test_one_failed_session_in_31_holds() -> None:
    """The round-2 reviewer's case: 800 prompts, 31 sessions, one session in
    which the candidate answers everything wrong, policy 0.10. The exact
    one-sided 95% bound on the failure rate with one occurrence in 31 is
    0.144, which does not exclude 0.10. The submitted gate passed it."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_fixture(work, repeats=31, common=800, rare=0)
        path = work / "candidate_r31.jsonl"
        recs = records(path)
        for rec in recs:
            rec["output_text"] = json.dumps({"answer": "WRONG"})
        write_records(path, recs)
        v, done = regate(
            work, quality=0.10, compliance=0.05, success=0.01, session_failure=0.10
        )
    aq = {d["metric"]: d for d in v["decisions"]}["answer_quality"]
    print(
        f"  release {v['verdict']} exit {done.returncode}; answer_quality "
        f"{aq['verdict']} delta {aq['mean_delta']:+.4f} "
        f"[{aq['ci_lo']:+.4f}, {aq['ci_hi']:+.4f}]; failed sessions "
        f"{aq['session_failures_observed']}, rate upper {aq['session_rate_upper']:.4f}, "
        f"sessions needed {aq['sessions_needed']}"
    )
    assert aq["session_failures_observed"] == 1, aq
    assert abs(aq["session_rate_upper"] - 0.14409) < 5e-5, aq
    assert aq["sessions_needed"] == 46, aq
    assert aq["verdict"] == "INCONCLUSIVE", aq
    assert (v["verdict"], done.returncode) == ("INCONCLUSIVE", 2), v["verdict"]
    assert "answer_quality" in v["held_by"], v["held_by"]


def test_mixed_persistent_and_outage_harm_holds() -> None:
    """Round-4 reviewer's mixed-harm case, the realization the gate passes.

    800 prompts; the candidate is persistently wrong on 56 of them in every
    session (a 7-point loss), and the true process also fails a whole session
    with probability 4% (every prompt wrong). The true mean quality delta is
    -(56/800) - 0.04*(744/800) = -0.1072, past the -0.10 margin. But in the
    realized 29-session sample no outage landed, so the observed mean is only
    -0.07 and the observed failure count is 0. The frequency guard clears
    (0 outages bound the rate at 0.098 < 0.10) and, before this fix, so did
    the mean: the gate certified a -0.1072 change as non-inferior. The mean
    claim must account for the unobserved severe-session tail: with the
    failure rate bounded only at 0.098 and a failure session worth as much as
    -1, the mean could be as low as (1-0.098)*(-0.07) + 0.098*(-1) = -0.161,
    which the -0.10 margin does not cover. INCONCLUSIVE, not PASS.
    """
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        write_fixture(work, repeats=29, common=800, rare=0)
        for r in range(1, 30):
            path = work / f"candidate_r{r}.jsonl"
            recs = records(path)
            for rec in recs:
                if int(rec["id"][1:]) < 56:  # persistently wrong on 56 prompts
                    rec["output_text"] = json.dumps({"answer": "WRONG"})
            write_records(path, recs)
        v, done = regate(
            work, quality=0.10, compliance=0.05, success=0.01, session_failure=0.10
        )
    aq = {d["metric"]: d for d in v["decisions"]}["answer_quality"]
    print(
        f"  release {v['verdict']} exit {done.returncode}; answer_quality "
        f"{aq['verdict']} observed delta {aq['mean_delta']:+.4f}; failed "
        f"sessions {aq['session_failures_observed']}, rate upper "
        f"{aq['session_rate_upper']:.3f}, mean lower bound "
        f"{aq['session_mean_lb']:+.4f}"
    )
    # The observed mean is the mild -0.07; the true mean -0.1072 is not in the
    # sample, so this asserts on the honest lower bound, not the point estimate.
    assert aq["session_failures_observed"] == 0, aq
    assert abs(aq["mean_delta"] + 0.07) < 1e-9, aq
    assert aq["session_mean_lb"] < -0.10, aq
    assert aq["verdict"] == "INCONCLUSIVE", aq
    assert (v["verdict"], done.returncode) == ("INCONCLUSIVE", 2), v["verdict"]
    assert "answer_quality" in v["held_by"], v["held_by"]


def main() -> None:
    print("case 1: one unresolved family among an all-PASS aggregate, 42 sessions")
    test_unresolved_family_holds_release()
    print("case 2: a duplicate request id inside one run is rejected")
    test_duplicate_id_rejected()
    print("case 3: a record missing from one run is rejected")
    test_missing_record_rejected()
    print("case 4: a run whose records carry another run's id is rejected")
    test_mismatched_run_id_rejected()
    print("case 5: failed requests stay failed observations and block")
    test_failed_requests_are_observations()
    print("case 6: copying every record into five sessions narrows nothing")
    test_copies_do_not_narrow()
    print("case 7: a stale PASS on disk does not survive a crashed re-gate")
    test_stale_verdict_cannot_survive_crash()
    print("case 8: the verdict is bound to input hashes, policy and revision")
    test_report_is_bound_to_inputs()
    print("case 9: a session with no latency is ERROR, never a verdict")
    test_nonfinite_measurement_is_error()
    print("case 10: a release needs a declared policy; diagnostic never approves")
    test_release_requires_a_declared_policy()
    print("case 11: one failed session in 31 does not clear a 0.10 margin")
    test_one_failed_session_in_31_holds()
    print("case 12: persistent loss plus an unobserved outage tail holds the mean")
    test_mixed_persistent_and_outage_harm_holds()
    print("\nrelease verdict propagation: all cases behaved as planted")


if __name__ == "__main__":
    main()
