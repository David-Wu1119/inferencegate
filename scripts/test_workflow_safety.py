"""Static safety policy for the GPU example; Ruby's YAML parser is a test tool only.

    python scripts/test_workflow_safety.py ci/example-workflow.yml
    python scripts/test_workflow_safety.py --self-test ci/example-workflow.yml

--self-test also derives known-unsafe variants from the template and requires
every one of them to FAIL, so the policy cannot silently weaken.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PR_CODE = re.compile(r"refs/pull/|github\.head_ref|github\.event\.pull_request")


def faults(path):
    data = json.loads(
        subprocess.check_output(
            [
                "ruby",
                "-rjson",
                "-ryaml",
                "-e",
                "puts JSON.generate(YAML.safe_load(File.read(ARGV[0])))",
                str(path),
            ],
            text=True,
        )
    )
    errors = []
    if data.get("on") != {"workflow_dispatch": None}:
        errors.append("trigger must be manual-only, without caller-supplied inputs")
    if data.get("permissions") != {"contents": "read"}:
        errors.append("token must have contents:read only")
    jobs = data.get("jobs", {})
    if set(jobs) != {"qualify"}:
        errors.append("unexpected job outside the guarded qualifier")
    job = jobs.get("qualify", {})
    # A job-level block replaces the workflow-level token permissions.
    if job.get("permissions", {"contents": "read"}) != {"contents": "read"}:
        errors.append("job must not widen token permissions")
    guard = " ".join(job.get("if", "").split())
    expected = (
        "github.event_name == 'workflow_dispatch' && "
        "github.event.repository.private == true && "
        "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)"
    )
    if guard != expected:
        errors.append(
            "job must reject PR events, public repositories and non-default branches before runner scheduling"
        )
    steps = job.get("steps", [])
    # Action owners and names resolve case-insensitively on GitHub.
    checkouts = [
        s for s in steps if s.get("uses", "").lower().startswith("actions/checkout@")
    ]
    if len(checkouts) != 1 or checkouts[0].get("with") != {
        "ref": "${{ github.sha }}",
        "persist-credentials": False,
    }:
        errors.append(
            "checkout must pin the event SHA and must not persist credentials"
        )
    if any(PR_CODE.search(str(s.get("run", ""))) for s in steps):
        errors.append("run steps must not fetch or reference pull-request code")
    return errors


# (name, text to replace, replacement): each result must FAIL the policy.
UNSAFE = [
    (
        "pull_request trigger",
        "  workflow_dispatch:\n",
        "  workflow_dispatch:\n  pull_request:\n",
    ),
    ("guard removed", "    if: >-\n", "    if-disabled: >-\n"),
    ("job-level write-all", "    runs-on:", "    permissions: write-all\n    runs-on:"),
    (
        "job-level contents write",
        "    runs-on:",
        "    permissions:\n      contents: write\n    runs-on:",
    ),
    ("head_ref checkout", "ref: ${{ github.sha }}", "ref: ${{ github.head_ref }}"),
    (
        "persisted credentials",
        "persist-credentials: false",
        "persist-credentials: true",
    ),
    (
        "second checkout, other case",
        "      - name: Bring up",
        "      - uses: Actions/checkout@v4\n        with:\n          ref: refs/pull/1/head\n      - name: Bring up",
    ),
    (
        "run step fetches PR code",
        "        run: ./serving/up.sh",
        "        run: git fetch origin refs/pull/1/head && git checkout FETCH_HEAD && ./serving/up.sh",
    ),
]


def self_test(path):
    text = Path(path).read_text()
    failures = [] if not faults(path) else [f"shipped template fails: {faults(path)}"]
    with tempfile.TemporaryDirectory() as tmp:
        for name, old, new in UNSAFE:
            if text.count(old) != 1:
                failures.append(
                    f"{name}: template no longer contains the mutation anchor {old!r}"
                )
                continue
            variant = Path(tmp) / "variant.yml"
            variant.write_text(text.replace(old, new))
            if not faults(variant):
                failures.append(f"{name}: unsafe variant passed the policy")
    print(
        json.dumps(
            {
                "file": str(path),
                "unsafe_variants": len(UNSAFE),
                "result": "FAIL" if failures else "PASS",
                "failures": failures,
            },
            indent=2,
        )
    )
    return failures


if __name__ == "__main__":
    if sys.argv[1] == "--self-test":
        raise SystemExit(bool(self_test(sys.argv[2])))
    errors = faults(sys.argv[1])
    print(
        json.dumps(
            {
                "file": sys.argv[1],
                "result": "FAIL" if errors else "PASS",
                "faults": errors,
            },
            indent=2,
        )
    )
    raise SystemExit(bool(errors))
