"""Qualify every declared change axis and return one release verdict.

A release rarely changes one thing. An upgrade moves the engine version, and
the same window often carries a new quantization, a different GPU pool, or a
decoding-method switch. This reads a manifest describing the baseline and each
candidate axis, qualifies every axis, and returns the worst verdict, so a
deploy is gated on the whole change set rather than on whichever piece somebody
remembered to benchmark.

    python3 qualify_matrix.py --manifest release.json --out runs/

Two kinds of axis are supported:

``endpoint``
    Replay a workload against a baseline URL and a candidate URL. Covers
    engine version, model, GPU, quantization and serving configuration,
    because all of them are observable as "a different server answering the
    same requests".

``command``
    Run an arbitrary measurement command that prints a JSON object with
    ``baseline`` and ``candidate`` arrays of per-item pass/fail outcomes.
    Covers axes with no server to point at, such as a kernel or decoding
    change verified in-process.

Exit status is the verdict: 0 PASS, 1 BLOCK, 2 INCONCLUSIVE.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from release_gate import (  # noqa: E402
    BLOCK,
    ERROR,
    INCONCLUSIVE,
    PASS,
    combine,
    decide_paired_grid,
)

HERE = Path(__file__).resolve().parent
EXIT = {PASS: 0, BLOCK: 1, INCONCLUSIVE: 2, ERROR: 3}


def failed(message: str) -> dict:
    """An axis whose measurement did not complete. ERROR, never a decision."""
    return {
        "verdict": ERROR,
        "decisions": [],
        "blocked_by": [],
        "held_by": [],
        "error": message,
    }


KNOWN_AXES = {
    "engine_version",
    "model",
    "gpu",
    "quantization",
    "kernel",
    "decoding_method",
    "serving_config",
}


def session_policy(axis: dict, manifest: dict) -> tuple[float | None, bool]:
    """The session-risk policy an axis is qualified under.

    ``diagnostic: true`` at the top level makes the whole release an
    observed-sessions diagnostic that reports but cannot approve. Otherwise
    every axis needs a session-failure margin, its own or the manifest's.
    """
    diagnostic = bool(manifest.get("diagnostic"))
    margin = axis.get("session_failure_margin", manifest.get("session_failure_margin"))
    return (None if diagnostic else margin), diagnostic


def qualify_endpoint(axis: dict, manifest: dict, out: Path) -> dict:
    """Replay the workload against baseline and candidate URLs."""
    margin, diagnostic = session_policy(axis, manifest)
    cmd = [
        sys.executable,
        str(HERE / "qualify.py"),
        "--baseline-url",
        manifest["baseline"]["url"],
        "--candidate-url",
        axis["url"],
        "--model",
        axis.get("model", manifest["baseline"].get("model", "")),
        "--baseline-model",
        manifest["baseline"].get("model", ""),
        "--candidate-model",
        axis.get("model", manifest["baseline"].get("model", "")),
        "--requests",
        manifest["requests"],
        "--out",
        str(out),
        "--repeats",
        str(axis.get("repeats", manifest.get("repeats", 5))),
    ]
    for flag, key in (
        ("--arrival-rate", "arrival_rate"),
        ("--max-tokens", "max_tokens"),
        ("--slo-ttft-ms", "slo_ttft_ms"),
        ("--slo-tpot-ms", "slo_tpot_ms"),
        ("--expect", "expect"),
        ("--engine", "engine"),
        # Margins are policy and belong in the manifest beside the axis they
        # govern. Without passthrough an axis silently inherits defaults its
        # own sample size may not support, and reports inconclusive for a
        # reason the manifest never stated.
        ("--margin-goodput-rps", "margin_goodput_rps"),
        ("--margin-throughput-rps", "margin_throughput_rps"),
        ("--margin-quality", "margin_quality"),
        ("--margin-compliance", "margin_compliance"),
        ("--margin-success", "margin_success"),
        ("--margin-p95-ttft-ms", "margin_p95_ttft_ms"),
        ("--margin-p95-tpot-ms", "margin_p95_tpot_ms"),
        ("--margin-cost", "margin_cost"),
    ):
        value = axis.get(key, manifest.get(key))
        if value is not None:
            cmd += [flag, str(value)]
    # The session policy is forwarded exactly; qualify.py refuses to run
    # without one, so an axis can never be qualified under a policy the
    # report does not carry.
    cmd += ["--diagnostic"] if diagnostic else ["--session-failure-margin", str(margin)]
    # Whatever an earlier run left here is not this run's verdict.
    out.mkdir(parents=True, exist_ok=True)
    verdict_path = out / "verdict.json"
    verdict_path.unlink(missing_ok=True)
    done = subprocess.run(cmd, capture_output=True, text=True)
    if not verdict_path.exists():
        return failed(f"qualify.py produced no verdict: {done.stderr[-600:]}")
    try:
        return json.loads(verdict_path.read_text())
    except ValueError as exc:
        return failed(f"qualify.py wrote an unreadable verdict: {exc}")


def qualify_command(axis: dict, manifest: dict, out: Path) -> dict:
    """Run a measurement command once per session and gate the paired outcomes.

    The command prints a JSON object with ``baseline`` and ``candidate``
    arrays of per-item pass/fail outcomes. It is run ``repeats`` times (the
    axis's, else the manifest's, else once); each run is a session, so the
    declared session policy applies to command axes exactly as it does to
    endpoint axes. A single-session command cannot bound a session-failure
    rate under any useful margin and is INCONCLUSIVE under one.
    """
    margin, diagnostic = session_policy(axis, manifest)
    repeats = int(axis.get("repeats", manifest.get("repeats", 1)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").unlink(missing_ok=True)
    grid: dict[str, list[tuple[bool, bool]]] = {}
    for r in range(repeats):
        done = subprocess.run(
            axis["command"],
            shell=True,
            capture_output=True,
            text=True,
            cwd=axis.get("cwd"),
        )
        if done.returncode != 0:
            return failed(
                f"measurement command failed (session {r + 1}): {done.stderr[-600:]}"
            )
        try:
            payload = json.loads(done.stdout.strip().splitlines()[-1])
            baseline, candidate = payload["baseline"], payload["candidate"]
            if len(baseline) != len(candidate):
                raise ValueError("baseline and candidate differ in length")
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            return failed(
                f"command printed no usable JSON result (session {r + 1}, {exc}): "
                f"{done.stdout[-400:]}"
            )
        if r and len(baseline) != len(grid):
            return failed(
                f"session {r + 1} reported {len(baseline)} items, earlier sessions "
                f"{len(grid)}; sessions must measure the same items"
            )
        for k, (b, c) in enumerate(zip(baseline, candidate, strict=True)):
            grid.setdefault(f"item{k:06d}", []).append((bool(b), bool(c)))
    d = decide_paired_grid(
        grid,
        margin=axis.get("margin", 0.02),
        metric=axis.get("metric", "success_rate"),
        unit="item",
        session_failure_margin=margin,
        diagnostic=diagnostic,
    )
    result = {
        "verdict": d.verdict,
        "scope": "observed-sessions-diagnostic" if diagnostic else "release",
        "policy": {"session_failure_margin": margin, "diagnostic": diagnostic},
        "blocked_by": [d.metric] if d.verdict == BLOCK else [],
        "held_by": [d.metric] if d.verdict == INCONCLUSIVE else [],
        "decisions": [d.__dict__],
    }
    (out / "verdict.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # The release report is this run's output and nothing else, so whatever
    # an earlier run left here goes before anything can fail. The 2026-09-05
    # review reproduced the alternative: a manifest rejected below left an
    # old PASS standing, and the CI wrapper read it as this run's result.
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "release_verdict.json").unlink(missing_ok=True)
    try:
        manifest = json.loads(args.manifest.read_text())
        candidates = manifest["candidates"]
        if not isinstance(candidates, list):
            raise TypeError("candidates is not a list")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A manifest that cannot be read measured nothing. Exit 3, not the
        # traceback's exit 1 that a shell would read as BLOCK.
        print(
            f"error: cannot read the release manifest {args.manifest}: {exc}",
            file=sys.stderr,
        )
        return EXIT[ERROR]

    unknown = {a.get("axis") for a in candidates} - KNOWN_AXES
    if unknown:
        raise SystemExit(
            f"unknown change axes {sorted(unknown)}; declare one of "
            f"{sorted(KNOWN_AXES)} so the report says what actually changed"
        )
    # A release manifest must declare the session-risk policy it is
    # qualified under, or say it is a diagnostic. Missing configuration is
    # ERROR before anything runs; insufficient evidence is INCONCLUSIVE later.
    diagnostic = bool(manifest.get("diagnostic"))
    if not diagnostic:
        unpoliced = [
            a["name"]
            for a in candidates
            if a.get("session_failure_margin", manifest.get("session_failure_margin"))
            is None
        ]
        if unpoliced:
            print(
                f"error: the release manifest declares no session-failure margin "
                f"for {unpoliced}: set session_failure_margin at the top level or "
                f"on every axis (the largest per-session probability of a "
                f"whole-session regression the release must exclude), or set "
                f"diagnostic: true for an analysis that cannot approve",
                file=sys.stderr,
            )
            return EXIT[ERROR]

    results = []
    for axis in manifest["candidates"]:
        name = axis["name"]
        print(f"\n>>> qualifying {name} (axis: {axis['axis']})", file=sys.stderr)
        out = args.out / name
        if axis.get("kind", "endpoint") == "command":
            res = qualify_command(axis, manifest, out)
        else:
            res = qualify_endpoint(axis, manifest, out)
        results.append({"name": name, "axis": axis["axis"], **res})

    overall = combine(r["verdict"] for r in results)

    print("\n" + "=" * 78)
    print(f"Release qualification across {len(results)} change axes")
    print("=" * 78)
    for r in results:
        detail = ""
        if r["blocked_by"]:
            detail = "blocked by " + ", ".join(r["blocked_by"])
        elif r["held_by"]:
            detail = "held by " + ", ".join(r["held_by"])
        if r.get("error"):
            detail = r["error"].strip().splitlines()[-1][:70]
        print(f"  {r['verdict']:13} {r['name']:24} [{r['axis']:16}] {detail}")
    print("-" * 78)
    print(f"RELEASE VERDICT: {overall}")
    if diagnostic:
        print("  scope: observed-sessions diagnostic; this cannot approve a release")
    if overall == ERROR:
        print(
            "  qualification did not complete on at least one axis; fix the "
            "measurement before reading this as a decision about the change"
        )
    elif overall == BLOCK:
        print("  do not deploy this change set")
    elif overall == INCONCLUSIVE:
        print("  do not deploy yet: at least one axis is unmeasured or unresolved")

    (args.out / "release_verdict.json").write_text(
        json.dumps(
            {
                "verdict": overall,
                "scope": "observed-sessions-diagnostic" if diagnostic else "release",
                "policy": {
                    "session_failure_margin": manifest.get("session_failure_margin"),
                    "diagnostic": diagnostic,
                },
                "axes": results,
            },
            indent=2,
        )
    )
    return EXIT[overall]


if __name__ == "__main__":
    raise SystemExit(main())
