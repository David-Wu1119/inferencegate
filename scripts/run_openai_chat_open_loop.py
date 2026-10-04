#!/usr/bin/env python3
"""Open-loop (Poisson-arrival) driver for the quality-aware serving harness.

run_openai_chat_batch.py is closed-loop: it keeps a fixed number of requests in
flight (max_workers = concurrency). That answers "how does the engine behave at
a fixed in-flight count," which a reviewer can call unrealistic. This driver is
open-loop: requests arrive at a target rate lambda (requests/sec) drawn from a
Poisson process, regardless of how many are already in flight, so the server's
own queue -- not an artificial client cap -- shapes TTFT/TPOT. This is the E3
realism upgrade.

It reuses the request-building and HTTP logic from run_openai_chat_batch so
there is one source of truth for payloads and latency measurement. Output rows
match the existing result schema (so score_quality.py and the SLO sweeps work
unchanged) plus three open-loop fields: target_rate_qps, arrival_offset_s, and
submit_lag_ms (how late the client dispatched the arrival -- a client-side
saturation check that should stay near zero).

Dependency-free; threads (not processes) because arrivals share one timeline and
post_json is I/O-bound.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_openai_chat_batch as base  # noqa: E402  (shared request/HTTP logic)


def poisson_arrival_offsets(n: int, rate_qps: float, rng: random.Random) -> list[float]:
    """Cumulative arrival times (seconds from start) for a Poisson process."""
    offsets: list[float] = []
    t = 0.0
    for _ in range(n):
        # inter-arrival ~ Exponential(rate); rng.expovariate(rate) = -ln(U)/rate
        t += rng.expovariate(rate_qps)
        offsets.append(t)
    return offsets


def trace_arrival_offsets(path: Path, n: int, scale: float = 1.0) -> list[float]:
    """Cumulative arrival times from a real trace file: one inter-arrival time
    (seconds) per line, or a CSV whose last numeric field is the inter-arrival
    time. The trace's burstiness shape is replayed (scaled by `scale`); cycles
    the trace if it has fewer than n entries. Enables S1 (production-trace replay,
    e.g. BurstGPT / the Azure LLM inference trace) without changing the engine path."""
    vals: list[float] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.replace(",", " ").split()
        try:
            vals.append(float(tok[-1]))
        except ValueError:
            continue
    if not vals:
        raise ValueError(f"no numeric inter-arrival values in {path}")
    offsets: list[float] = []
    t = 0.0
    for i in range(n):
        t += vals[i % len(vals)] * scale
        offsets.append(t)
    return offsets


def _pipeline_check(text: str, schema):
    """extract -> L2 repair -> validate, exactly as measure_repair_pipeline scores it.
    Returns (valid, error_message_or_None)."""
    sq = importlib.import_module("score_quality")
    mrp = importlib.import_module("measure_repair_pipeline")
    ok, obj = sq.parse_json_strict(text or "")
    if not ok:
        ok, obj = sq.extract_json_object(text or "")
    if not ok:
        return False, "no JSON object could be extracted from the response"
    if not schema:
        return False, "no schema"
    valid, err = sq.validate_schema(schema, mrp.repair(obj, schema, 2))
    return bool(valid), (None if valid else str(err))


def _pipeline_valid(text: str, schema) -> bool:
    return _pipeline_check(text, schema)[0]


def _feedback_row(row, prev_text: str, err: str):
    """The retry prompt: original task plus the previous invalid output and the
    validator's complaint. Kept short so the retry is not a different workload."""
    base_prompt = row.get("condition_prompt") or row["prompt"]
    prev = (prev_text or "")[:1500]
    fb = (
        f"{base_prompt}\n\nYour previous response was not valid for the required JSON schema.\n"
        f"Validator error: {err}\nPrevious response:\n{prev}\n"
        f"Return only a corrected JSON object that satisfies the schema."
    )
    r2 = dict(row)
    r2["condition_prompt"] = fb
    return r2


def _retry_until_valid(args, row, record, started_unix):
    """The competing pipeline, issued rather than estimated. Each extra attempt is
    a genuine second request submitted at once, so it competes with the arrivals
    still in flight. Effective latency runs from the ORIGINAL arrival to the end of
    the final attempt; TTFT_effective is time-to-first-token of the attempt whose
    output is kept, measured from the original arrival, so a retried request pays
    its first attempt in full. TPOT is the kept attempt's own."""
    schema = row.get("schema")
    attempts = [record]
    valid, err = (
        _pipeline_check(record.get("output_text"), schema)
        if not record.get("error")
        else (False, record.get("error"))
    )
    first_valid = valid
    while not valid and len(attempts) - 1 < args.retry_max:
        retry_row = (
            _feedback_row(row, attempts[-1].get("output_text"), err)
            if args.retry_feedback
            else row
        )
        nxt = base.run_one(args, retry_row)
        attempts.append(nxt)
        valid, err = (
            _pipeline_check(nxt.get("output_text"), schema)
            if not nxt.get("error")
            else (False, nxt.get("error"))
        )
    final = attempts[-1]
    out = dict(final)
    out["attempts"] = len(attempts)
    out["first_attempt_valid"] = first_valid
    out["final_valid"] = valid
    out["retry_feedback"] = bool(args.retry_feedback)
    out["attempts_detail"] = [
        {
            "started": a.get("request_started_at_unix"),
            "completed": a.get("request_completed_at_unix"),
            "latency": a.get("latency"),
            "usage": a.get("usage"),
            "error": a.get("error"),
            "output_text": a.get("output_text"),
            "finish_reason": a.get("finish_reason"),
        }
        for a in attempts
    ]
    out["request_started_at_unix"] = started_unix
    lat = dict(final.get("latency") or {})
    if len(attempts) > 1:
        lat["end_to_end_ms"] = (
            final.get("request_completed_at_unix", started_unix) - started_unix
        ) * 1000.0
        fin_ttft = (final.get("latency") or {}).get("ttft_ms")
        if fin_ttft is not None and final.get("request_started_at_unix") is not None:
            lat["ttft_ms"] = (
                final["request_started_at_unix"] - started_unix
            ) * 1000.0 + fin_ttft
        lat["first_attempt"] = attempts[0].get("latency")
    out["latency"] = lat
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", required=True, help="Example: http://127.0.0.1:8000/v1"
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    parser.add_argument(
        "--condition",
        required=True,
        choices=["free_text", "prompt_json", "constrained_json"],
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--arrival-rate",
        required=False,
        type=float,
        default=None,
        help="Target Poisson arrival rate lambda, in requests/sec.",
    )
    parser.add_argument(
        "--arrival-trace",
        type=Path,
        default=None,
        help="Replay a real arrival trace (inter-arrival seconds per line) "
        "instead of Poisson; see S1 in the pre-registration.",
    )
    parser.add_argument(
        "--trace-scale",
        type=float,
        default=1.0,
        help="Multiply trace inter-arrival times (e.g. compress the trace to a higher rate).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Arrival-process seed.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle request order before assigning arrivals.",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=2048,
        help="Client thread cap; keep >> expected in-flight to stay open-loop.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument(
        "--stream", action="store_true", help="Use streaming to estimate TTFT."
    )
    parser.add_argument(
        "--no-stream-include-usage", action="store_false", dest="stream_include_usage"
    )
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="Implement the competing pipeline under load: after each response, "
        "extract -> schema-aware repair (L2) -> validate against the row's schema; "
        "if still invalid, ISSUE a real second request immediately (it competes "
        "with in-flight arrivals). Records attempts, first_attempt_valid, "
        "final_valid, and effective latency from arrival to the final attempt.",
    )
    parser.add_argument(
        "--retry-max", type=int, default=1, help="Extra attempts allowed (default 1)."
    )
    parser.add_argument(
        "--retry-feedback",
        action="store_true",
        help="Feedback-augmented retry: the retry prompt carries the previous invalid "
        "output and the validator's error, so a greedy model is not asked the "
        "identical question. Without this, a temperature-0 retry mostly reproduces "
        "the same invalid output.",
    )
    parser.set_defaults(stream_include_usage=True)
    args = parser.parse_args()

    rows = base.read_jsonl(args.requests)
    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(rows)
    if args.arrival_trace is not None:
        offsets = trace_arrival_offsets(args.arrival_trace, len(rows), args.trace_scale)
    elif args.arrival_rate is not None:
        offsets = poisson_arrival_offsets(len(rows), args.arrival_rate, rng)
    else:
        parser.error(
            "provide either --arrival-rate (Poisson) or --arrival-trace (replay)"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = [None] * len(rows)  # type: ignore[list-item]

    def dispatch(
        idx: int, row: dict[str, Any], arrival_offset: float, start_wall: float
    ) -> None:
        submit_lag_ms = (
            max(0.0, (time.perf_counter() - start_wall) - arrival_offset) * 1000.0
        )
        started = time.time()
        try:
            record = base.run_one(args, row)
            if args.retry_invalid:
                record = _retry_until_valid(args, row, record, started)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            # Defence in depth. run_one already converts a refused request
            # into a record carrying an "error" field, so this path only fires
            # if it raises before its own handler, such as a malformed row
            # failing payload construction. Recording rather than dropping
            # keeps that case out of the missing-record abort below, since a
            # bad row is data about the workload, not a lost measurement.
            print(f"request {row.get('id')!r} failed: {exc}", file=sys.stderr)
            results[idx] = {
                "id": row.get("id"),
                "error": f"{type(exc).__name__}: {exc}",
                "model": args.model,
                "engine": args.engine,
                "condition": args.condition,
                "run_id": args.run_id,
                "request_started_at_unix": started,
                "request_completed_at_unix": time.time(),
                "latency": {},
                "usage": {},
                "output_text": None,
                "target_rate_qps": args.arrival_rate,
                "arrival_offset_s": arrival_offset,
                "submit_lag_ms": submit_lag_ms,
                "arrival_seed": args.seed,
            }
            return
        record["target_rate_qps"] = args.arrival_rate
        record["arrival_offset_s"] = arrival_offset
        record["submit_lag_ms"] = submit_lag_ms
        record["arrival_seed"] = args.seed
        results[idx] = record

    start_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
        for idx, (row, offset) in enumerate(zip(rows, offsets, strict=False)):
            now = time.perf_counter() - start_wall
            sleep_for = offset - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            pool.submit(dispatch, idx, row, offset, start_wall)
        # ThreadPoolExecutor context exit waits for all in-flight requests.

    missing = sum(1 for record in results if record is None)
    if missing:
        # A slot is only empty if dispatch itself failed to record anything,
        # which is an apparatus fault rather than a server error: a refused
        # request now leaves an error record. Writing the survivors and
        # exiting 0 would report a partial run as a healthy one, which is the
        # failure mode this harness exists to detect.
        raise SystemExit(
            f"{missing}/{len(results)} requests produced no record; "
            f"refusing to write a run that silently lost measurements"
        )

    with args.out.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    # Provenance manifest, same writer as the closed-loop runner: engine/library
    # versions, GPU UUID, host, and token-count coverage. Without it the
    # --same-gpu audit reports "unrecorded" and a pre-registered block voids
    # itself. Open-loop has no worker pool; record the arrival model instead.
    args.concurrency = args.max_inflight
    args.worker_mode = "thread-openloop"
    manifest = base.write_run_manifest(args, results)
    manifest["arrival"] = {
        "rate_qps": args.arrival_rate,
        "trace": str(args.arrival_trace) if args.arrival_trace else None,
        "seed": args.seed,
        "retry_invalid": bool(args.retry_invalid),
        "retry_max": args.retry_max,
        "retry_feedback": bool(args.retry_feedback),
    }
    (args.out.parent / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    offered = args.arrival_rate
    span = offsets[-1] if offsets else 0.0
    achieved = len(rows) / span if span > 0 else float("nan")
    max_lag = max((r.get("submit_lag_ms", 0.0) for r in results if r), default=0.0)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "n": len(rows),
                "offered_rate_qps": offered,
                "arrival_span_s": round(span, 3),
                "nominal_achieved_arrival_qps": round(achieved, 3)
                if not math.isnan(achieved)
                else None,
                "max_client_submit_lag_ms": round(max_lag, 1),
                "note": "max_client_submit_lag_ms >> 0 means the client could not keep up; raise --max-inflight",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
