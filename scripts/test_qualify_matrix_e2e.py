"""End-to-end check of the multi-axis driver, with no GPU.

Builds release manifests against mock endpoints and throwaway measurement
commands, runs the real driver, and asserts the release verdict and exit code.
The cases that matter most are the last three: a measurement that crashes or
prints garbage must surface as ERROR and outrank BLOCK, because a broken
adapter that reads as "the gate said no" is a policy decision nobody made.

    identical endpoint arms              -> PASS,  exit 0
    candidate answers 40% wrong          -> BLOCK, exit 1
    command axis, healthy outcomes       -> PASS
    command axis, degraded outcomes      -> BLOCK
    crashing command beside a BLOCK axis -> ERROR, exit 3, ERROR wins
    command that prints no JSON          -> ERROR
    undeclared change axis               -> rejected before anything runs
    one prompt alone in its family       -> INCONCLUSIVE, exit 2, family named
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mock_endpoint import serve  # noqa: E402
from test_qualify_e2e import REQUESTS  # noqa: E402

DRIVER = HERE / "qualify_matrix.py"


def write_command(workdir: Path, name: str, body: str) -> str:
    """A measurement command is any script whose last stdout line is JSON."""
    path = workdir / f"{name}.py"
    path.write_text(body)
    return f"{sys.executable} {path}"


def run_manifest(workdir: Path, name: str, manifest: dict):
    path = workdir / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=2))
    out = workdir / name
    done = subprocess.run(
        [sys.executable, str(DRIVER), "--manifest", str(path), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    report = out / "release_verdict.json"
    verdict = json.loads(report.read_text()) if report.exists() else None
    if verdict is None:
        # A missing report is either the planted outcome or a crash; show
        # the driver's last words either way so a crash cannot hide.
        print(
            f"  (no report; driver exit {done.returncode}; stderr: {done.stderr[-600:]!r})"
        )
    return done.returncode, verdict, done


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        reqs = work / "requests.jsonl"
        reqs.write_text("\n".join(json.dumps(r) for r in REQUESTS) + "\n")

        base = dict(
            cv=0.10,
            output_tokens=8,
            error_rate=0.0,
            wrong_rate=0.0,
            noncompliant_rate=0.0,
            ttft_ms=30,
            tpot_ms=6,
        )
        serve(8981, seed=1, **base)
        serve(8982, seed=2, **base)
        serve(8983, seed=3, **{**base, "wrong_rate": 0.4})
        time.sleep(0.4)

        # Five repeats, like the single-axis suite: at three the t-interval on a
        # run-level metric is 55% wider (t=4.30 at df=2 against 2.78 at df=4)
        # and a throughput delta of -0.1 rps on a mock straddles a 1 rps
        # margin, so the identical-arms case reads INCONCLUSIVE. That is the
        # gate being right about too few runs, not a fault to relax the margin
        # around. 80 prompts with zero observed disagreement settle margins
        # above 3/80 = 0.0375 by the rule of three; repeats do not lower it.
        common = {
            "requests": str(reqs),
            "repeats": 5,
            # The declared session policy. At 0.10 a PASS needs 29 sessions
            # (35 at the two-family alpha of case 8); the BLOCK/ERROR cases
            # need only five, since they never reach the session check.
            "session_failure_margin": 0.10,
            "arrival_rate": 40,
            "max_tokens": 8,
            "slo_ttft_ms": 500,
            "slo_tpot_ms": 30,
            "expect": "ANSWER_OK",
            # Wide quality margin for the mechanical suite; the strict 0.10
            # mean-quality session budget is asserted in test_release_verdict.
            "margin_quality": 0.30,
            "margin_compliance": 0.10,
            "margin_success": 0.10,
            "margin_p95_tpot_ms": 5,
            # See test_qualify_e2e: two seconds of makespan make 1 rps a coin
            # flip; nothing planted here regresses throughput.
            "margin_goodput_rps": 3.0,
            "margin_throughput_rps": 3.0,
            "baseline": {
                "url": "http://127.0.0.1:8981/v1",
                "model": "mock",
                "name": "current",
            },
        }

        print("case 1: identical endpoint arms, 29 sessions")
        code, v, _ = run_manifest(
            work,
            "same",
            {
                **common,
                "repeats": 29,
                "candidates": [
                    {
                        "name": "same-build",
                        "axis": "engine_version",
                        "kind": "endpoint",
                        "url": "http://127.0.0.1:8982/v1",
                        "model": "mock",
                    }
                ],
            },
        )
        print(f"  release {v['verdict']}, exit {code}")
        assert v is not None, "case 1 wrote no report"
        assert v["axes"][0]["policy"]["session_failure_margin"] == 0.10, v["axes"][0]
        assert v["scope"] == "release", v["scope"]
        assert (code, v["verdict"]) == (0, "PASS"), (
            code,
            v["verdict"],
            v["axes"][0]["held_by"],
        )

        print("case 2: candidate answers 40% of items wrong")
        code, v, _ = run_manifest(
            work,
            "wrong",
            {
                **common,
                "candidates": [
                    {
                        "name": "degraded-build",
                        "axis": "quantization",
                        "kind": "endpoint",
                        "url": "http://127.0.0.1:8983/v1",
                        "model": "mock",
                    }
                ],
            },
        )
        blocked = [
            d["metric"] for d in v["axes"][0]["decisions"] if d["verdict"] == "BLOCK"
        ]
        print(f"  release {v['verdict']}, exit {code}, blocked by {blocked}")
        assert (code, v["verdict"]) == (1, "BLOCK"), (code, v)
        assert "answer_quality" in blocked, blocked

        healthy = write_command(
            work,
            "healthy",
            (
                "import json\n"
                "print(json.dumps({'baseline': [True]*60, 'candidate': [True]*60}))\n"
            ),
        )
        degraded = write_command(
            work,
            "degraded",
            (
                "import json\n"
                "print(json.dumps({'baseline': [True]*60,"
                " 'candidate': [True]*30 + [False]*30}))\n"
            ),
        )
        crashing = write_command(
            work,
            "crashing",
            ("import sys\nprint('probe exploded', file=sys.stderr)\nsys.exit(1)\n"),
        )
        silent = write_command(work, "silent", "print('no json here')\n")

        print("case 3: command axis, healthy outcomes, 29 sessions")
        code, v, _ = run_manifest(
            work,
            "cmd-ok",
            {
                **common,
                "candidates": [
                    {
                        "name": "kernel-v2",
                        "axis": "kernel",
                        "kind": "command",
                        "command": healthy,
                        "metric": "parse_survival",
                        "margin": 0.10,
                        "repeats": 29,
                    }
                ],
            },
        )
        print(f"  release {v['verdict']}, exit {code}")
        assert (code, v["verdict"]) == (0, "PASS"), (code, v)
        assert v["axes"][0]["decisions"][0]["repeats"] == 29, v["axes"][0]["decisions"][
            0
        ]

        print("case 4: command axis, degraded outcomes")
        code, v, _ = run_manifest(
            work,
            "cmd-bad",
            {
                **common,
                "candidates": [
                    {
                        "name": "kernel-v3",
                        "axis": "kernel",
                        "kind": "command",
                        "command": degraded,
                        "metric": "parse_survival",
                        "margin": 0.10,
                    }
                ],
            },
        )
        print(f"  release {v['verdict']}, exit {code}")
        assert (code, v["verdict"]) == (1, "BLOCK"), (code, v)

        print("case 5: crashing measurement beside a blocking axis")
        code, v, _ = run_manifest(
            work,
            "crash",
            {
                **common,
                "candidates": [
                    {
                        "name": "degraded-build",
                        "axis": "quantization",
                        "kind": "endpoint",
                        "url": "http://127.0.0.1:8983/v1",
                        "model": "mock",
                    },
                    {
                        "name": "broken-probe",
                        "axis": "decoding_method",
                        "kind": "command",
                        "command": crashing,
                    },
                ],
            },
        )
        by_name = {a["name"]: a for a in v["axes"]}
        print(
            f"  release {v['verdict']}, exit {code}; "
            f"degraded-build={by_name['degraded-build']['verdict']} "
            f"broken-probe={by_name['broken-probe']['verdict']}"
        )
        assert (code, v["verdict"]) == (3, "ERROR"), (code, v)
        assert by_name["degraded-build"]["verdict"] == "BLOCK", by_name
        assert by_name["broken-probe"]["verdict"] == "ERROR", by_name
        assert "probe exploded" in by_name["broken-probe"]["error"], by_name

        print("case 6: measurement prints no JSON")
        code, v, _ = run_manifest(
            work,
            "silent",
            {
                **common,
                "candidates": [
                    {
                        "name": "mute-probe",
                        "axis": "decoding_method",
                        "kind": "command",
                        "command": silent,
                    }
                ],
            },
        )
        print(f"  release {v['verdict']}, exit {code}")
        assert (code, v["verdict"]) == (3, "ERROR"), (code, v)

        print("case 7: undeclared change axis")
        code, v, done = run_manifest(
            work,
            "unknown",
            {
                **common,
                "candidates": [
                    {
                        "name": "mystery",
                        "axis": "vibes",
                        "kind": "command",
                        "command": healthy,
                    }
                ],
            },
        )
        print(f"  exit {code}, verdict file written: {v is not None}")
        assert code != 0 and v is None, (code, v)
        assert "unknown change axes" in done.stderr, done.stderr[-300:]

        # A family holding a single prompt cannot be resolved at any margin.
        # The 2026-09-05 audit found such a family reported and then ignored,
        # so the release read PASS. It has to hold the release, and the
        # summary has to name what held it. Sixty prompts in the common
        # family keep the aggregate quality metric PASS whether quality is
        # counted over observations or over prompts, so this case cannot pass
        # by everything going INCONCLUSIVE.
        print("case 8: one prompt alone in its family, identical arms")
        rows = [
            {
                **REQUESTS[0],
                "id": f"c{i:02d}",
                "prompt": f"common question {i}",
                "task_family": "common",
            }
            for i in range(60)
        ] + [
            {
                **REQUESTS[0],
                "id": "r00",
                "prompt": "the only question in its family",
                "task_family": "rare",
            }
        ]
        rare_reqs = work / "rare_requests.jsonl"
        rare_reqs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        # 36 sessions: the two families are gated at alpha 0.025, where 36
        # agreeing sessions bound the failure rate at 0.097 (35 leave 0.10003),
        # so the common family and the aggregate clear the 0.10 policy while
        # the lone rare prompt still holds the release for lack of prompts.
        code, v, _ = run_manifest(
            work,
            "rare",
            {
                **common,
                "requests": str(rare_reqs),
                "repeats": 36,
                "candidates": [
                    {
                        "name": "same-build",
                        "axis": "engine_version",
                        "kind": "endpoint",
                        "url": "http://127.0.0.1:8982/v1",
                        "model": "mock",
                    }
                ],
            },
        )
        axis = v["axes"][0]
        by_metric = {d["metric"]: d for d in axis["decisions"]}
        by_family = {d["metric"]: d for d in axis["family_decisions"]}
        print(
            f"  release {v['verdict']}, exit {code}; answer_quality "
            f"{by_metric['answer_quality']['verdict']}, quality[common] "
            f"{by_family['quality[common]']['verdict']}, quality[rare] "
            f"{by_family['quality[rare]']['verdict']}"
        )
        assert by_metric["answer_quality"]["verdict"] == "PASS", by_metric
        assert by_family["quality[common]"]["verdict"] == "PASS", by_family
        assert by_family["quality[rare]"]["verdict"] == "INCONCLUSIVE", by_family
        assert (code, v["verdict"]) == (2, "INCONCLUSIVE"), (code, v["verdict"])
        assert axis["verdict"] == "INCONCLUSIVE", axis["verdict"]
        assert "quality[rare]" in axis["held_by"], axis.get("held_by")

        # A crashed qualification must not inherit a verdict from an earlier
        # run in the same output directory. Both the release report and the
        # axis report are pre-seeded with PASS; the axis then points at a
        # workload file that does not exist, so qualify.py cannot start.
        print("case 9: stale PASS reports on disk, then the qualification crashes")
        stale = work / "stale"
        (stale / "ghost-build").mkdir(parents=True)
        (stale / "release_verdict.json").write_text(
            json.dumps({"verdict": "PASS", "axes": []})
        )
        (stale / "ghost-build" / "verdict.json").write_text(
            json.dumps(
                {"verdict": "PASS", "decisions": [], "blocked_by": [], "held_by": []}
            )
        )
        code, v, done = run_manifest(
            work,
            "stale",
            {
                **common,
                "requests": str(work / "does-not-exist.jsonl"),
                "candidates": [
                    {
                        "name": "ghost-build",
                        "axis": "engine_version",
                        "kind": "endpoint",
                        "url": "http://127.0.0.1:8982/v1",
                        "model": "mock",
                    }
                ],
            },
        )
        print(f"  release {v['verdict'] if v else None}, exit {code}")
        assert v is not None and (code, v["verdict"]) == (3, "ERROR"), (code, v)
        assert v["axes"][0]["verdict"] == "ERROR", v["axes"][0]
        assert "does-not-exist" in v["axes"][0]["error"], v["axes"][0]["error"]

        print("case 10: a malformed manifest is ERROR, not a decision")
        bad = work / "bad.json"
        bad.write_text("{not json")
        done = subprocess.run(
            [
                sys.executable,
                str(DRIVER),
                "--manifest",
                str(bad),
                "--out",
                str(work / "bad"),
            ],
            capture_output=True,
            text=True,
        )
        print(
            f"  exit {done.returncode}, report written: {(work / 'bad' / 'release_verdict.json').exists()}"
        )
        assert done.returncode == 3, (done.returncode, done.stderr[-300:])
        assert not (work / "bad" / "release_verdict.json").exists()
        assert "manifest" in done.stderr, done.stderr[-300:]

        # The same bypass one layer down: a stale PASS release report plus a
        # manifest the driver rejects before it runs anything. The stale report
        # must be gone, not returned.
        print("case 11: stale PASS release report, then a malformed manifest")
        reused = work / "reused"
        reused.mkdir()
        (reused / "release_verdict.json").write_text(
            json.dumps({"verdict": "PASS", "axes": []})
        )
        bad2 = work / "bad2.json"
        bad2.write_text(json.dumps({"candidates": "invalid"}))
        done = subprocess.run(
            [
                sys.executable,
                str(DRIVER),
                "--manifest",
                str(bad2),
                "--out",
                str(reused),
            ],
            capture_output=True,
            text=True,
        )
        report = reused / "release_verdict.json"
        print(
            f"  exit {done.returncode}, stale report still present: {report.exists()}"
        )
        assert done.returncode == 3, (done.returncode, done.stderr[-300:])
        assert not report.exists(), report.read_text() if report.exists() else None

        print(
            "case 12: a manifest with no session policy is ERROR before anything runs"
        )
        nopol = {k: v for k, v in common.items() if k != "session_failure_margin"}
        code, v, done = run_manifest(
            work,
            "nopolicy",
            {
                **nopol,
                "candidates": [
                    {
                        "name": "kernel-v2",
                        "axis": "kernel",
                        "kind": "command",
                        "command": healthy,
                    }
                ],
            },
        )
        print(f"  exit {code}, report written: {v is not None}")
        assert code == 3 and v is None, (code, v)
        assert "session_failure_margin" in done.stderr, done.stderr[-300:]

    print("\nqualify_matrix end-to-end: all cases behaved as planted")


if __name__ == "__main__":
    main()
