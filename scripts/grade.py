"""Grade a model response against an expected answer.

Substring grading is the obvious implementation and it is wrong in a way that
matters for a release gate: an expected ``7`` is contained in ``17``, so a
candidate that answers every arithmetic item off by ten still grades perfect.
The quality metric then reads as non-inferior and the gate passes a real
regression.

So structured extraction is tried first. Workload items declare an
``answer_path``, prompts ask for JSON, and a response that yields that field is
compared for equality after normalization. Only a response with no usable
structure falls back to a text search, and that search is word-boundary
anchored rather than a bare substring. Callers are told which path was used,
because a run graded mostly by fallback is a run whose quality number should
not be trusted.
"""

from __future__ import annotations

import json
import re

STRUCTURED = "structured"
FALLBACK = "fallback"
UNGRADED = "ungraded"


def _normalize(value: str) -> str:
    """Casefold and drop punctuation and spacing, keeping sign digits letters."""
    return re.sub(r"[^a-z0-9\-]+", "", str(value).strip().lower())


def _json_blocks(text: str):
    """Yield candidate JSON objects in *text*, widest first."""
    stripped = text.strip()
    if stripped:
        yield stripped
    # Fenced blocks and prose-wrapped objects both leave a bare {...} behind.
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : i + 1]


def extract_answer(text: str, answer_path: str = "answer") -> str | None:
    """Return the declared answer field from *text*, or None if absent."""
    if not text:
        return None
    for block in _json_blocks(text):
        try:
            obj = json.loads(block)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and answer_path in obj:
            value = obj[answer_path]
            if isinstance(value, (str, int, float, bool)):
                return str(value)
    return None


def grade(text: str, expected: str, answer_path: str = "answer") -> tuple[bool, str]:
    """Grade one response.

    Returns:
        ``(correct, how)`` where *how* is ``structured``, ``fallback`` or
        ``ungraded``, so a caller can report the provenance of its own
        quality metric.
    """
    if expected is None:
        return False, UNGRADED
    want = _normalize(expected)
    found = extract_answer(text, answer_path)
    if found is not None:
        return _normalize(found) == want, STRUCTURED
    if not text:
        return False, UNGRADED
    # Word-boundary search so 7 does not match 17 and "no" does not match "not".
    pattern = re.compile(
        rf"(?<![a-z0-9]){re.escape(expected.strip().lower())}(?![a-z0-9])"
    )
    return bool(pattern.search(text.lower())), FALLBACK


def demo() -> None:
    """Self-check on the cases that make substring grading unsafe."""
    # Structured extraction, both string and numeric JSON values.
    assert grade('{"answer": "4"}', "4") == (True, STRUCTURED)
    assert grade('{"answer": 4}', "4") == (True, STRUCTURED)
    assert grade('{\n  "answer": "Paris"\n}', "paris") == (True, STRUCTURED)

    # The bug this module exists to remove: 7 must not match 17.
    assert grade('{"answer": "17"}', "7") == (False, STRUCTURED)
    assert grade("the answer is 17", "7") == (False, FALLBACK)
    assert grade("the answer is 7.", "7") == (True, FALLBACK)

    # A wrong structured answer is wrong even if the text mentions the right one.
    assert grade('{"answer": "10-3"}', "7") == (False, STRUCTURED)

    # Prose-wrapped and fenced JSON still grade structurally.
    assert (
        grade('Sure! ```json\n{"answer": "Tokyo"}\n``` done', "Tokyo")[1] == STRUCTURED
    )
    assert grade('Sure! {"answer": "Tokyo"} done', "Tokyo") == (True, STRUCTURED)

    # No structure at all falls back, and empty output is ungraded.
    assert grade("Tokyo", "Tokyo") == (True, FALLBACK)
    assert grade("", "Tokyo") == (False, UNGRADED)

    # Case and punctuation are not the model's fault.
    assert grade('{"answer": " ODD "}', "odd") == (True, STRUCTURED)

    # The declared field name is honoured, not assumed.
    assert grade('{"result": "4"}', "4", answer_path="result") == (True, STRUCTURED)
    assert grade('{"result": "4"}', "4", answer_path="answer")[1] == FALLBACK

    # A nested object in the answer field is not a scalar answer, so grading
    # falls back to text search and reports that it did.
    assert grade('{"answer": {"city": "Tokyo"}}', "Paris") == (False, FALLBACK)

    # Structured extraction wins even when the text also contains the answer
    # elsewhere, so a wrong field cannot be rescued by surrounding prose.
    assert grade('Tokyo is nice. {"answer": "Kyoto"}', "Tokyo") == (False, STRUCTURED)

    print("grade self-check: all cases pass")


if __name__ == "__main__":
    demo()
