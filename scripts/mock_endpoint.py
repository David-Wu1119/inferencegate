"""A tiny OpenAI-compatible endpoint for exercising the release gate offline.

The qualification pipeline needs two servers to compare. Renting two GPUs to
prove the pipeline works is the wrong order of operations, so this stands in:
it speaks enough of ``/v1/chat/completions`` (streaming and not) for the replay
harness, and its latency, error rate and answer quality are set from the
command line. That makes it possible to plant a known regression and check the
gate reaches the verdict it should.

    python3 mock_endpoint.py --port 8801 --ttft-ms 40 --tpot-ms 8
    python3 mock_endpoint.py --port 8802 --ttft-ms 40 --tpot-ms 26  # regressed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _jitter(mean_ms: float, cv: float, rng: random.Random) -> float:
    """Positive lognormal-ish jitter so runs differ the way real ones do."""
    if mean_ms <= 0:
        return 0.0
    sd = max(1e-6, mean_ms * cv)
    return max(0.0, rng.gauss(mean_ms, sd))


class Handler(BaseHTTPRequestHandler):
    """Per-server config lives on the subclass, never on a module global.

    Two mock endpoints run in one process during the end-to-end test, so a
    shared global would let the second server silently overwrite the first
    and make both arms identical, which reads as "no regression found".
    """

    protocol_version = "HTTP/1.1"
    cfg: dict = {}

    def log_message(self, *_args):  # keep the harness output readable
        pass

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cfg = self.cfg
        # Per-request outcome is a function of (server seed, request body), so the
        # same request fails or succeeds identically across repeats and pairs by
        # id. The digest must be stable across processes: Python's hash() of a
        # str is randomized per interpreter, which made the planted error rate
        # land as 25 failures in one process and 5 in another.
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).digest()
        rng = random.Random(cfg["seed"] ^ int.from_bytes(digest[:4], "big"))

        if rng.random() < cfg["error_rate"]:
            self.send_error(503, "mock overload")
            return

        n_tokens = cfg["output_tokens"]
        good = rng.random() >= cfg["wrong_rate"]
        answer = "ANSWER_OK" if good else "ANSWER_WRONG"
        # Real prompts ask for a JSON answer field, so the stand-in returns one
        # too. noncompliant_rate drops the envelope while keeping the answer
        # itself intact, which is how a format regression looks in the wild:
        # the model still knows, it just stops honouring the response contract.
        if rng.random() < cfg.get("noncompliant_rate", 0.0):
            text = answer
        else:
            text = json.dumps({"answer": answer})
        pieces = [text] + [f" t{i}" for i in range(n_tokens - 1)]

        time.sleep(_jitter(cfg["ttft_ms"], cfg["cv"], rng) / 1000.0)

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i, piece in enumerate(pieces):
                if i:
                    time.sleep(_jitter(cfg["tpot_ms"], cfg["cv"], rng) / 1000.0)
                self._chunk({"choices": [{"index": 0, "delta": {"content": piece}}]})
            self._chunk(
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 32,
                        "completion_tokens": n_tokens,
                        "total_tokens": 32 + n_tokens,
                    },
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return

        time.sleep(_jitter(cfg["tpot_ms"] * (n_tokens - 1), cfg["cv"], rng) / 1000.0)
        self._json(
            {
                "id": "mock",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(pieces)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 32,
                    "completion_tokens": n_tokens,
                    "total_tokens": 32 + n_tokens,
                },
            }
        )

    def _chunk(self, payload: dict) -> None:
        data = b"data: " + json.dumps(payload).encode() + b"\n\n"
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _json(self, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve(port: int, **cfg) -> ThreadingHTTPServer:
    handler = type(f"Handler{port}", (Handler,), {"cfg": dict(cfg)})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ttft-ms", type=float, default=40.0)
    ap.add_argument("--tpot-ms", type=float, default=8.0)
    ap.add_argument("--cv", type=float, default=0.15, help="run-to-run spread")
    ap.add_argument("--output-tokens", type=int, default=24)
    ap.add_argument("--error-rate", type=float, default=0.0)
    ap.add_argument("--wrong-rate", type=float, default=0.0)
    ap.add_argument(
        "--noncompliant-rate",
        type=float,
        default=0.0,
        help="fraction of responses that omit the JSON envelope",
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    serve(
        a.port,
        ttft_ms=a.ttft_ms,
        tpot_ms=a.tpot_ms,
        cv=a.cv,
        output_tokens=a.output_tokens,
        error_rate=a.error_rate,
        wrong_rate=a.wrong_rate,
        noncompliant_rate=a.noncompliant_rate,
        seed=a.seed,
    )
    print(f"mock endpoint on http://127.0.0.1:{a.port}/v1")
    threading.Event().wait()


if __name__ == "__main__":
    main()
