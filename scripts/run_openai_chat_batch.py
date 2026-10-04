#!/usr/bin/env python3
"""Run a JSONL request batch against an OpenAI-compatible chat endpoint.

This is intentionally dependency-free so it can run on a fresh pod. It is a
T0 harness, not the final benchmark client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def build_prompt(row: dict[str, Any], condition: str) -> str:
    prompt = row["prompt"]
    if condition in {"prompt_json", "constrained_json"}:
        prompt = row.get("condition_prompt") or prompt
        keys = row.get("expected", {}).get("json_required_keys") or []
        key_text = ", ".join(keys) if keys else "answer"
        return (
            f"{prompt}\n\n"
            f"Return only a compact JSON object with these required keys: {key_text}."
        )
    return prompt


def default_schema(row: dict[str, Any]) -> dict[str, Any]:
    keys = row.get("expected", {}).get("json_required_keys") or ["answer"]
    properties = {}
    required = []
    for key in keys:
        required.append(key)
        if key == "label":
            properties[key] = {"type": "string"}
        else:
            properties[key] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_payload(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": build_prompt(row, args.condition)}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": args.n,
        "stream": args.stream,
    }
    if args.stream and args.stream_include_usage:
        payload["stream_options"] = {"include_usage": True}
    if args.condition == "constrained_json":
        schema = row.get("schema") or default_schema(row)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "quality_aware_serving_t0",
                "schema": schema,
                "strict": True,
            },
        }
    return payload


def post_json(
    url: str, api_key: str, payload: dict[str, Any], stream: bool
) -> tuple[str, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")

    start = time.perf_counter()
    first_token_at: float | None = None
    output_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            if stream:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(f"stream error: {chunk['error']}")
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece and first_token_at is None:
                        first_token_at = time.perf_counter()
                    output_parts.append(piece)
            else:
                parsed = json.loads(response.read().decode("utf-8"))
                if parsed.get("error"):
                    raise RuntimeError(f"response error: {parsed['error']}")
                choices = parsed.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    output_parts.append(message.get("content") or "")
                    finish_reason = choices[0].get("finish_reason")
                usage = parsed.get("usage") or {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc

    end = time.perf_counter()
    output_text = "".join(output_parts)
    output_tokens = usage.get("completion_tokens")
    latency: dict[str, Any] = {"end_to_end_ms": (end - start) * 1000.0}
    if first_token_at is not None:
        latency["ttft_ms"] = (first_token_at - start) * 1000.0
        if output_tokens and output_tokens > 1:
            latency["tpot_ms"] = ((end - first_token_at) * 1000.0) / (output_tokens - 1)
            latency["itl_ms"] = latency["tpot_ms"]

    normalized_usage = {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": output_tokens,
    }
    return output_text, {
        "latency": latency,
        "usage": normalized_usage,
        "finish_reason": finish_reason,
    }


def run_one(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    payload = build_payload(args, row)
    url = args.base_url.rstrip("/") + "/chat/completions"
    request_started_at_unix = time.time()
    try:
        output_text, metrics = post_json(url, args.api_key, payload, args.stream)
        request_completed_at_unix = time.time()
        return {
            "id": row["id"],
            "run_id": args.run_id,
            "engine": args.engine,
            "condition": args.condition,
            "model": args.model,
            "request_started_at_unix": request_started_at_unix,
            "request_completed_at_unix": request_completed_at_unix,
            "output_text": output_text,
            **metrics,
        }
    except Exception as exc:  # Preserve per-request failures in the result log.
        request_completed_at_unix = time.time()
        return {
            "id": row.get("id"),
            "run_id": args.run_id,
            "engine": args.engine,
            "condition": args.condition,
            "model": args.model,
            "request_started_at_unix": request_started_at_unix,
            "request_completed_at_unix": request_completed_at_unix,
            "output_text": "",
            "error": str(exc),
        }


def _pkg_versions(names: list[str]) -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _gpu_info() -> list[dict[str, Any]]:
    """Includes the GPU UUID: without it two runs cannot be shown to have used
    the same physical accelerator, which is the whole premise of a
    fixed-hardware repeat."""
    import subprocess

    try:
        raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in raw.strip().splitlines():
        parts = [f.strip() for f in line.split(",")]
        if len(parts) == 4:
            gpus.append(
                {
                    "name": parts[0],
                    "memory": parts[1],
                    "driver": parts[2],
                    "uuid": parts[3],
                }
            )
    return gpus


def _cpu_model() -> str | None:
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _engine_version(base_url: str, api_key: str) -> str | None:
    """Best effort: vLLM serves /version; SGLang serves /get_server_info."""
    root = base_url.rstrip("/").removesuffix("/v1")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for path, key in (("/version", "version"), ("/get_server_info", "version")):
        try:
            req = urllib.request.Request(root + path, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict) and data.get(key):
                return str(data[key])
        except Exception:
            continue
    return None


def _requests_info(path) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    return {"path": str(p), "sha256": digest}


def write_run_manifest(
    args: argparse.Namespace, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Record what produced this run. Without this a row cannot be attributed
    to a configuration after the fact, which silently invalidates any
    cross-run comparison (see the tpot_ms provenance guard below)."""
    n = len(rows)
    ok = sum(1 for r in rows if not r.get("error"))
    with_usage = sum(
        1 for r in rows if isinstance((r.get("usage") or {}).get("output_tokens"), int)
    )
    with_tpot = sum(
        1 for r in rows if (r.get("latency") or {}).get("tpot_ms") is not None
    )
    manifest = {
        "run_id": args.run_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": args.engine,
        "engine_version": _engine_version(args.base_url, args.api_key),
        "model": args.model,
        "condition": args.condition,
        # The requests file is the treatment in the matched-information block: record which one.
        "requests": _requests_info(getattr(args, "requests", None)),
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n": args.n,
        },
        "load": {"concurrency": args.concurrency, "worker_mode": args.worker_mode},
        "streaming": {
            "stream": args.stream,
            "stream_include_usage": args.stream_include_usage,
        },
        "library_versions": _pkg_versions(
            ["vllm", "sglang", "transformers", "torch", "fastapi", "starlette"]
        ),
        # The client spawns `concurrency` workers on this host, so an
        # under-provisioned CPU shows up as inflated TTFT on the server's numbers.
        "hardware": {
            "gpus": _gpu_info(),
            "cpu_count": os.cpu_count(),
            "cpu_model": _cpu_model(),
            "workers_per_cpu": round(args.concurrency / (os.cpu_count() or 1), 2),
        },
        "python": sys.version.split()[0],
        "dataset": {"requests": str(args.requests), "request_count": n},
        "files": {"results": str(args.out)},
        "counts": {
            "requests": n,
            "completed": ok,
            "with_engine_output_tokens": with_usage,
            "with_measured_tpot_ms": with_tpot,
        },
        "tpot_provenance": "engine" if with_tpot == n else "incomplete",
    }
    path = args.out.parent / "run_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--worker-mode", choices=["process", "thread"], default="process"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Use streaming to estimate TTFT."
    )
    parser.add_argument(
        "--no-stream-include-usage", action="store_false", dest="stream_include_usage"
    )
    parser.add_argument(
        "--allow-reconstructed-tpot",
        action="store_false",
        dest="require_measured_tpot",
        help="Accept a run whose TPOT must be reconstructed offline from re-tokenized "
        "text. Such a run is NOT comparable with engine-measured runs.",
    )
    parser.set_defaults(stream_include_usage=True, require_measured_tpot=True)
    args = parser.parse_args()

    rows = read_jsonl(args.requests)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    executor_cls = (
        ProcessPoolExecutor if args.worker_mode == "process" else ThreadPoolExecutor
    )
    results: list[dict[str, Any]] = []
    with args.out.open("w", encoding="utf-8") as handle:
        with executor_cls(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_one, args, row) for row in rows]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                handle.write(json.dumps(result, sort_keys=True) + "\n")
                handle.flush()

    manifest = write_run_manifest(args, results)
    counts = manifest["counts"]
    if args.stream and args.require_measured_tpot:
        missing = counts["requests"] - counts["with_measured_tpot_ms"]
        if missing:
            print(
                f"ERROR: {missing}/{counts['requests']} requests have no engine-measured "
                "tpot_ms. Downstream scoring would silently reconstruct TPOT from a "
                "re-tokenization of the output text, which is biased and not comparable "
                "across runs. Ensure the server honours stream_options.include_usage, "
                "or pass --allow-reconstructed-tpot to accept the bias knowingly.",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
