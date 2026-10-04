"""End-to-end check of the qualification pipeline, with no GPU.

Stands up two mock endpoints, plants a known difference between them, runs the
real replay-reduce-gate pipeline, and asserts the verdict.

    identical arms, 80 prompts x 29   -> PASS (wide quality margin; the strict
                                         0.10 mean-quality session cost is
                                         asserted in test_release_verdict)
    candidate 4x slower TPOT          -> BLOCK on p95_tpot_ms
    candidate answers wrong           -> BLOCK on quality, not on latency
    candidate drops the JSON envelope -> BLOCK on schema_compliance
    candidate fails a fifth           -> BLOCK on request_success
    identical arms, 16 prompts        -> INCONCLUSIVE: too few prompts to
                                         settle a 0.05 margin, however many
                                         times each is replayed
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

REQUESTS = [
    # json_required_keys is what makes the response format part of the
    # contract, and therefore gateable rather than merely observable.
    {
        "id": f"q{i}",
        "prompt": f'question {i}. Reply with only JSON: {{"answer": "<value>"}}',
        "expected": {
            "answer": "ANSWER_OK",
            "answer_path": "answer",
            "json_required_keys": ["answer"],
        },
        "task_family": "demo",
        "metadata": {},
        "schema": None,
        "condition_prompt": None,
    }
    for i in range(80)
]


def run_case(
    name, base_port, cand_port, base_cfg, cand_cfg, repeats, workdir, requests=None
):
    defaults = dict(
        cv=0.10, output_tokens=8, error_rate=0.0, wrong_rate=0.0, noncompliant_rate=0.0
    )
    serve(base_port, seed=1, **{**defaults, **base_cfg})
    serve(cand_port, seed=2, **{**defaults, **cand_cfg})
    time.sleep(0.4)

    reqs = workdir / f"{name}_requests.jsonl"
    reqs.write_text("\n".join(json.dumps(r) for r in requests or REQUESTS) + "\n")

    done = subprocess.run(
        [
            sys.executable,
            str(HERE / "qualify.py"),
            "--baseline-url",
            f"http://127.0.0.1:{base_port}/v1",
            "--candidate-url",
            f"http://127.0.0.1:{cand_port}/v1",
            "--model",
            "mock",
            "--requests",
            str(reqs),
            "--out",
            str(workdir / name),
            "--repeats",
            str(repeats),
            "--arrival-rate",
            "40",
            "--max-tokens",
            "8",
            "--slo-ttft-ms",
            "500",
            "--slo-tpot-ms",
            "30",
            "--expect",
            "ANSWER_OK",
            "--margin-p95-tpot-ms",
            "5",
            # 80 requests at 40 rps replay in about two seconds of makespan, so
            # 0.1 s of scheduler jitter between arms is 2 rps of throughput. A
            # 1 rps margin on that is a coin flip on a loaded machine (the
            # round-2 clean-archive run on 3.11 held case 1 on it). 3 rps is
            # 7.5% of the rate and nothing planted here regresses throughput.
            "--margin-goodput-rps",
            "3",
            "--margin-throughput-rps",
            "3",
            # Wide quality margin: this suite tests which metric refuses each
            # planted regression, not the strict 0.10 mean-quality session
            # budget (that lives in test_release_verdict, saved-record, fast).
            # At 0.30 the identical-arms case still PASSes at 29 sessions while
            # the 50%-wrong and format cases still refuse on the right metric.
            "--margin-quality",
            "0.30",
            # 80 prompts with zero observed disagreement settle a margin above
            # 3/80 = 0.0375 by the rule of three; repeats do not lower that
            # bound, because they replay the same prompts.
            "--margin-compliance",
            "0.05",
            "--margin-success",
            "0.05",
            # The declared session policy: at 0.10 a PASS needs 29 sessions.
            "--session-failure-margin",
            "0.10",
        ],
        capture_output=True,
        text=True,
    )
    verdict = json.loads((workdir / name / "verdict.json").read_text())
    return verdict, done


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        print("case 1: identical arms, 29 sessions")
        v, _ = run_case(
            "same",
            8901,
            8902,
            {"ttft_ms": 30, "tpot_ms": 6},
            {"ttft_ms": 30, "tpot_ms": 6},
            29,
            work,
        )
        print(f"  verdict {v['verdict']}")
        assert v["verdict"] == "PASS", (v["verdict"], v["blocked_by"], v["held_by"])

        print("case 2: candidate 4x slower per output token")
        v, _ = run_case(
            "slow",
            8903,
            8904,
            {"ttft_ms": 30, "tpot_ms": 6},
            {"ttft_ms": 30, "tpot_ms": 24},
            5,
            work,
        )
        tpot = [d for d in v["decisions"] if d["metric"] == "p95_tpot_ms"][0]
        print(
            f"  verdict {v['verdict']}, p95_tpot_ms {tpot['verdict']} "
            f"delta {tpot['mean_delta']:+.1f} ms"
        )
        assert v["verdict"] == "BLOCK", v["verdict"]
        assert tpot["verdict"] == "BLOCK", tpot

        print("case 3: candidate answers half the items wrong, same speed")
        v, _ = run_case(
            "wrong",
            8905,
            8906,
            {"ttft_ms": 30, "tpot_ms": 6},
            {"ttft_ms": 30, "tpot_ms": 6, "wrong_rate": 0.5},
            3,
            work,
        )
        quality = [d for d in v["decisions"] if d["metric"] == "answer_quality"][0]
        tpot = [d for d in v["decisions"] if d["metric"] == "p95_tpot_ms"][0]
        print(
            f"  verdict {v['verdict']}, answer_quality {quality['verdict']} "
            f"delta {quality['mean_delta']:+.2f}, p95_tpot_ms {tpot['verdict']}"
        )
        # The arms are configured identically for speed, so a latency metric
        # must not be what blocks the release; quality has to carry it.
        assert quality["verdict"] == "BLOCK", quality
        assert tpot["verdict"] != "BLOCK", tpot
        assert v["verdict"] == "BLOCK", v["verdict"]

        # A change can keep every answer right and still stop honouring the
        # response format. Quality does not block (at five sessions it holds
        # under the session policy rather than passing), so a quality-only
        # gate would never refuse this; schema_compliance is what refuses.
        print("case 4: candidate drops the JSON envelope on half the items")
        v, _ = run_case(
            "format",
            8907,
            8908,
            {"ttft_ms": 30, "tpot_ms": 6},
            {"ttft_ms": 30, "tpot_ms": 6, "noncompliant_rate": 0.5},
            5,
            work,
        )
        comp = [d for d in v["decisions"] if d["metric"] == "schema_compliance"][0]
        qual = [d for d in v["decisions"] if d["metric"] == "answer_quality"][0]
        print(
            f"  verdict {v['verdict']}, schema_compliance "
            f"{comp['verdict']} delta {comp['mean_delta']:+.2f}, "
            f"answer_quality {qual['verdict']}"
        )
        assert comp["verdict"] == "BLOCK", comp
        assert qual["verdict"] != "BLOCK", qual

        # Reliability. Before errors were recorded rather than dropped, a
        # candidate returning 503s aborted the replay, so the qualification
        # crashed instead of blocking and the error rate was unmeasurable.
        print("case 5: candidate fails a fifth of requests")
        v, done = run_case(
            "errors",
            8909,
            8910,
            {"ttft_ms": 30, "tpot_ms": 6},
            {"ttft_ms": 30, "tpot_ms": 6, "error_rate": 0.2},
            5,
            work,
        )
        succ = [d for d in v["decisions"] if d["metric"] == "request_success"][0]
        print(
            f"  verdict {v['verdict']}, request_success {succ['verdict']} "
            f"delta {succ['mean_delta']:+.2f}"
        )
        assert succ["verdict"] == "BLOCK", succ
        assert v["verdict"] == "BLOCK", v["verdict"]

        # The original fixture: 16 prompts, five repeats, identical arms. It
        # used to PASS because 80 pooled observations cleared the rule of
        # three at 3/80. Sixteen prompts replayed five times are still
        # sixteen prompts, and sixteen cannot settle a 0.05 margin.
        print("case 6: 16 prompts replayed five times, identical arms")
        v, done = run_case(
            "thin",
            8911,
            8912,
            {"ttft_ms": 30, "tpot_ms": 6},
            {"ttft_ms": 30, "tpot_ms": 6},
            5,
            work,
            requests=REQUESTS[:16],
        )
        qual = [d for d in v["decisions"] if d["metric"] == "answer_quality"][0]
        print(
            f"  verdict {v['verdict']}, exit {done.returncode}, answer_quality "
            f"{qual['verdict']} over {qual.get('unique_prompts')} prompts x "
            f"{qual.get('repeats')}; held by {v.get('held_by')}"
        )
        assert qual["verdict"] == "INCONCLUSIVE", qual
        assert v["verdict"] == "INCONCLUSIVE" and done.returncode == 2, v["verdict"]
        assert "answer_quality" in v["held_by"], v["held_by"]
        assert v["blocked_by"] == [], v["blocked_by"]

    print("\nqualify end-to-end: all cases behaved as planted")


if __name__ == "__main__":
    main()
