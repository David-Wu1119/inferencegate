"""Generate a deterministic, exactly-checkable eval set for release qualification.

The 12-item set this replaces was all trivial arithmetic, which made the quality
metric both noisy and easy: a 0.5B model gets almost all of it right, so the
paired bootstrap had almost no signal to work with and the gate returned
INCONCLUSIVE with an estimate of ~3675 items.

Two properties matter more than size:

*   **Exactly checkable.** Every answer is a short canonical string produced by
    the generator itself, and every prompt asks for it in a JSON ``answer``
    field, so grading is an equality test rather than a substring search.
    Substring grading on numeric answers passes ``17`` for an expected ``7``.
*   **Selected for discordant-pair yield, not difficulty.** A paired
    comparison learns only from items where the two arms disagree, so a family
    both arms always get right (a ceiling) or always get wrong (a floor)
    spends GPU time and contributes no information. Measured on the first
    200-item run, ``chemical_symbol`` (100% in both arms), ``string_reverse``
    (0% in both) and ``multi_step`` (2.5% baseline) yielded 7 discordant pairs
    out of 245 between them, so they are replaced here by families in the
    productive middle band. Yield is a property of the model under test, so
    re-calibrate with ``quality_by_family.py`` when the baseline changes.
*   **Spread across task families.** Quantization does not degrade every
    capability evenly, so a single-family set can miss a real regression
    entirely, and one family can even improve while others collapse.
"""

from __future__ import annotations

import argparse
import collections
import json
import random

WORDS = [
    "orbit",
    "candle",
    "puzzle",
    "marble",
    "ribbon",
    "walnut",
    "cactus",
    "tunnel",
    "velvet",
    "hammer",
    "jungle",
    "pepper",
    "silver",
    "wagon",
    "basket",
    "dragon",
    "flute",
    "garden",
    "helmet",
    "island",
    "lantern",
    "compass",
    "harvest",
    "mineral",
    "outpost",
    "quarry",
    "saddle",
    "thicket",
    "vinegar",
    "whistle",
    "anchor",
    "bramble",
    "cinder",
    "driftwood",
    "ember",
    "furrow",
    "granite",
    "hollow",
    "kettle",
    "lattice",
    "mitten",
    "nutmeg",
    "parcel",
    "rafter",
    "satchel",
    "trellis",
    "umber",
    "wicker",
]
CAPITALS = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Canada": "Ottawa",
    "Egypt": "Cairo",
    "Brazil": "Brasilia",
    "Kenya": "Nairobi",
    "Norway": "Oslo",
    "Peru": "Lima",
    "Cuba": "Havana",
    "Nepal": "Kathmandu",
    "Portugal": "Lisbon",
    "Austria": "Vienna",
    "Greece": "Athens",
    "Sweden": "Stockholm",
    "Poland": "Warsaw",
    "Ireland": "Dublin",
    "Morocco": "Rabat",
    "Vietnam": "Hanoi",
    "Chile": "Santiago",
    "Iceland": "Reykjavik",
    "Hungary": "Budapest",
    "Thailand": "Bangkok",
    "Finland": "Helsinki",
    "Jamaica": "Kingston",
    "Denmark": "Copenhagen",
    "Ghana": "Accra",
    "Bolivia": "Sucre",
    "Latvia": "Riga",
}
JSON_RULE = ' Reply with only JSON: {"answer": "<value>"}'


def build(rng: random.Random, n: int) -> list[dict]:
    """Build *n* items, cycling the families so the mix is even."""
    families = [
        _arithmetic,
        _comparison,
        _string_length,
        _string_letter,
        _alphabetical_first,
        _list_count,
        _list_max,
        _sequence_next,
        _parity,
        _capital,
        _sum_list,
        _nth_item,
    ]
    items = []
    seen: set[str] = set()
    exhausted: dict[str, int] = {}
    for i in range(n):
        family = families[i % len(families)]
        # Two identical items are one observation counted twice, so a duplicate
        # is dropped rather than padded. A family that cannot produce a fresh
        # item within the retry budget is reported instead of silently
        # repeating itself.
        for _ in range(200):
            prompt, answer, name = family(rng)
            if prompt not in seen:
                break
        else:
            exhausted[name] = exhausted.get(name, 0) + 1
            continue
        seen.add(prompt)
        items.append(
            {
                "id": f"e{len(items):03d}",
                "prompt": prompt + JSON_RULE,
                "expected": {
                    "answer": answer,
                    "answer_path": "answer",
                    "json_required_keys": ["answer"],
                },
                "task_family": name,
                "metadata": {},
                "schema": None,
                "condition_prompt": None,
            }
        )
    if exhausted:
        detail = ", ".join(f"{k} (+{v})" for k, v in sorted(exhausted.items()))
        print(f"note: families ran out of distinct items: {detail}")
    return items


def _arithmetic(rng):
    a, b = rng.randint(11, 89), rng.randint(11, 89)
    op = rng.choice(["+", "-", "*"])
    value = {"+": a + b, "-": a - b, "*": a * b}[op]
    return f"Compute {a} {op} {b}.", str(value), "arithmetic"


def _comparison(rng):
    a, b = rng.sample(range(100, 999), 2)
    return (f"Which number is larger, {a} or {b}?", str(max(a, b)), "comparison")


def _string_length(rng):
    w = rng.choice(WORDS)
    return f"How many letters are in the word '{w}'?", str(len(w)), "string_length"


def _string_letter(rng):
    w = rng.choice(WORDS)
    which = rng.choice(["first", "last"])
    return (
        f"What is the {which} letter of the word '{w}'?",
        w[0] if which == "first" else w[-1],
        "string_letter",
    )


def _alphabetical_first(rng):
    """Ordering without arithmetic.

    This slot previously held a letter-counting family, which measured 5.9%
    on the baseline: under the quality margin, so it could never evidence a
    regression, and its apparent +29.4% swing was a coin flip near the floor
    diluting the aggregate. Replaced rather than kept for the item count.
    """
    words = rng.sample(WORDS, 3)
    joined = ", ".join(words)
    return (
        f"Which of these words comes first alphabetically: {joined}?",
        sorted(words)[0],
        "alphabetical_first",
    )


def _list_count(rng):
    items = rng.sample(WORDS, rng.randint(3, 7))
    return (
        f"How many words are in this list: {', '.join(items)}?",
        str(len(items)),
        "list_count",
    )


def _list_max(rng):
    nums = rng.sample(range(10, 99), rng.randint(4, 6))
    joined = ", ".join(str(x) for x in nums)
    return (
        f"What is the largest number in this list: {joined}?",
        str(max(nums)),
        "list_max",
    )


def _sequence_next(rng):
    start, step = rng.randint(2, 20), rng.randint(2, 9)
    seq = [start + step * k for k in range(4)]
    joined = ", ".join(str(x) for x in seq)
    return (
        f"What number comes next in this sequence: {joined}?",
        str(seq[-1] + step),
        "sequence_next",
    )


def _parity(rng):
    n = rng.randint(100, 999)
    return (f"Is {n} even or odd?", "even" if n % 2 == 0 else "odd", "parity")


def _capital(rng):
    country = rng.choice(list(CAPITALS))
    return f"What is the capital city of {country}?", CAPITALS[country], "capital"


def _sum_list(rng):
    """Summing three two-digit numbers is harder than taking a maximum."""
    nums = [rng.randint(11, 89) for _ in range(3)]
    joined = ", ".join(str(x) for x in nums)
    return f"What is the sum of {joined}?", str(sum(nums)), "sum_list"


def _nth_item(rng):
    """Indexing into a list needs ordering without any arithmetic."""
    items = rng.sample(WORDS, rng.randint(4, 6))
    k = rng.randint(1, len(items))
    ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")
    return (
        f"What is the {ordinal} word in this list: {', '.join(items)}?",
        items[k - 1],
        "nth_item",
    )


def self_check() -> None:
    """The properties a release gate needs from its workload, asserted."""
    a = build(random.Random(20260903), 200)
    b = build(random.Random(20260903), 200)
    assert a == b, "the same seed must build the same set in any process"
    assert build(random.Random(1), 200) != a, "a different seed must differ"

    prompts = [x["prompt"] for x in a]
    assert len(prompts) == 200, len(prompts)
    assert len(set(prompts)) == 200, "duplicate prompts are one observation twice"
    assert [x["id"] for x in a] == [f"e{i:03d}" for i in range(200)], "ids"

    families = collections.Counter(x["task_family"] for x in a)
    assert len(families) == 12, families
    assert max(families.values()) - min(families.values()) <= 1, families

    for x in a:
        answer = x["expected"]["answer"]
        assert isinstance(answer, str) and answer.strip(), x
        assert x["expected"]["answer_path"] == "answer", x
        assert x["expected"]["json_required_keys"] == ["answer"], (
            "format must be declared or schema_compliance is not gated"
        )
        assert x["prompt"].endswith(JSON_RULE), x

    # Answers are exactly checkable: spot-check families against the prompt.
    for x in a:
        q, ans = x["prompt"], x["expected"]["answer"]
        if x["task_family"] == "parity":
            n = int(q.split("Is ")[1].split(" even")[0])
            assert ans == ("even" if n % 2 == 0 else "odd"), x
        elif x["task_family"] == "sum_list":
            nums = [int(t) for t in q.split("sum of ")[1].split("?")[0].split(", ")]
            assert ans == str(sum(nums)), x
        elif x["task_family"] == "alphabetical_first":
            words = q.split("alphabetically: ")[1].split("?")[0].split(", ")
            assert ans == sorted(words)[0], x

    print("make_eval_set self-check: all cases pass")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="assert reproducibility, uniqueness and balance, then exit",
    )
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.out:
        ap.error("--out is required unless --self-check is given")

    items = build(random.Random(args.seed), args.n)
    with open(args.out, "w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")

    families: dict[str, int] = {}
    for item in items:
        families[item["task_family"]] = families.get(item["task_family"], 0) + 1
    print(f"wrote {len(items)} items to {args.out}")
    for name, count in sorted(families.items()):
        print(f"  {name:16} {count}")


if __name__ == "__main__":
    main()
