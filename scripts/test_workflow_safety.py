"""Static safety policy for the GPU example; Ruby's YAML parser is a test tool only."""

import json
import subprocess
import sys


def faults(path):
    data = json.loads(subprocess.check_output([
        "ruby", "-rjson", "-ryaml", "-e",
        "puts JSON.generate(YAML.safe_load(File.read(ARGV[0])))", str(path),
    ], text=True))
    errors = []
    if data.get("on") != {"workflow_dispatch": None}:
        errors.append("trigger must be manual-only, without caller-supplied inputs")
    if data.get("permissions") != {"contents": "read"}:
        errors.append("token must have contents:read only")
    jobs = data.get("jobs", {})
    if set(jobs) != {"qualify"}:
        errors.append("unexpected job outside the guarded qualifier")
    job = jobs.get("qualify", {})
    guard = " ".join(job.get("if", "").split())
    expected = ("github.event_name == 'workflow_dispatch' && "
                "github.event.repository.private == true && "
                "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)")
    if guard != expected:
        errors.append("job must reject PR events, public repositories and non-default branches before runner scheduling")
    checkouts = [s for s in job.get("steps", []) if s.get("uses", "").startswith("actions/checkout@")]
    if len(checkouts) != 1 or checkouts[0].get("with") != {
        "ref": "${{ github.sha }}", "persist-credentials": False,
    }:
        errors.append("checkout must pin the event SHA and must not persist credentials")
    return errors


if __name__ == "__main__":
    errors = faults(sys.argv[1])
    print(json.dumps({"file": sys.argv[1], "result": "FAIL" if errors else "PASS", "faults": errors}, indent=2))
    raise SystemExit(bool(errors))
