"""The composite action's verdict handling, run for real with bash.

Extracts the ``run:`` block from ``ci/action.yml``, supplies inputs through environment variables as
GitHub does, and executes it against command-axis manifests whose
outcome is known. No runner, no GPU. What is checked is the contract the
README states: the job fails on BLOCK, on ERROR, on a missing report, and by
default on INCONCLUSIVE; the ``verdict`` output is the measured verdict; and
an explicit ``allow-inconclusive`` override lets the job pass while keeping
the verdict INCONCLUSIVE and saying that an override was applied. The
override must never relabel the measurement as PASS.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACTION = ROOT / "ci" / "action.yml"


def run_block() -> str:
    """The shell of the action's single step, dedented."""
    source = ACTION.read_text()
    for variable, expression in {
        "GATE_MANIFEST": "inputs.manifest",
        "GATE_OUT": "inputs.out",
        "GATE_ALLOW_INCONCLUSIVE": "inputs.allow-inconclusive",
        "GATE_ACTION_PATH": "github.action_path",
    }.items():
        assert f"{variable}: ${{{{ {expression} }}}}" in source, variable
    lines = source.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "run: |") + 1
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for ln in lines[start:]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) < indent:
            break
        body.append(ln[indent:])
    return "\n".join(body) + "\n"


def command(workdir: Path, name: str, body: str) -> str:
    path = workdir / f"{name}.py"
    path.write_text(body)
    return f"{sys.executable} {path}"


def run_action(workdir: Path, name: str, manifest: dict | None, allow: str = "false"):
    manifest_path = workdir / f"{name}.json"
    if manifest is not None:
        manifest_path.write_text(json.dumps(manifest))
    out = workdir / name
    outputs = workdir / f"{name}.outputs"
    summary = workdir / f"{name}.summary"
    outputs.write_text("")
    summary.write_text("")
    script = run_block()
    assert "${{" not in script, "unsubstituted expression in the action shell"
    done = subprocess.run(
        ["bash", "-c", script],
        cwd=workdir,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GATE_ACTION_PATH": str(ROOT / "ci"),
            "GATE_MANIFEST": str(manifest_path),
            "GATE_OUT": str(out),
            "GATE_ALLOW_INCONCLUSIVE": allow,
            "GITHUB_OUTPUT": str(outputs),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
    )
    kv = dict(
        line.split("=", 1) for line in outputs.read_text().splitlines() if "=" in line
    )
    return done.returncode, kv, summary.read_text(), done


def manifest_for(
    cmd: str,
    margin: float = 0.10,
    axis: str = "kernel",
    session_failure_margin: float | None = 0.5,
    repeats: int = 5,
    diagnostic: bool = False,
    axis_policy: float | None = None,
) -> dict:
    """A release manifest with a declared session policy.

    A command axis is run ``repeats`` times, each run a session. With a 0.5
    margin, five agreeing sessions bound the failure rate at 0.451 and a
    healthy command can PASS; nothing planted here depends on more.
    """
    manifest = {
        "repeats": repeats,
        "candidates": [
            {
                "name": "probe",
                "axis": axis,
                "kind": "command",
                "command": cmd,
                "metric": "parse_survival",
                "margin": margin,
            }
        ],
    }
    if axis_policy is not None:
        manifest["candidates"][0]["session_failure_margin"] = axis_policy
    if diagnostic:
        manifest["diagnostic"] = True
    elif session_failure_margin is not None:
        manifest["session_failure_margin"] = session_failure_margin
    return manifest


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        healthy = command(
            work,
            "healthy",
            "import json\nprint(json.dumps({'baseline': [True]*60, 'candidate': [True]*60}))\n",
        )
        degraded = command(
            work,
            "degraded",
            "import json\nprint(json.dumps({'baseline': [True]*60,"
            " 'candidate': [True]*30 + [False]*30}))\n",
        )
        # Five agreeing pairs settle nothing at a 0.10 margin: INCONCLUSIVE.
        thin = command(
            work,
            "thin",
            "import json\nprint(json.dumps({'baseline': [True]*5, 'candidate': [True]*5}))\n",
        )
        crashing = command(work, "crashing", "import sys\nsys.exit(1)\n")

        print("adversarial inputs: shell metacharacters remain literal paths")
        marker = work / "INJECTED"
        for payload in [
            "$(touch INJECTED)",
            "`touch INJECTED`",
            '"; touch INJECTED; #',
            "spaces ' quotes ; &",
        ]:
            code, kv, _, done = run_action(work, payload, manifest_for(healthy))
            assert (code, kv.get("verdict")) == (0, "PASS"), (code, kv, done.stderr)
            assert not marker.exists(), payload
            assert (work / payload / "release_verdict.json").is_file(), payload
        code, kv, _, _ = run_action(
            work, "hostile-override", manifest_for(thin), "$(touch INJECTED)"
        )
        assert (code, kv.get("overridden")) == (1, "false"), (code, kv)
        assert not marker.exists()

        print("case 1: PASS passes the job")
        code, kv, _, done = run_action(work, "pass", manifest_for(healthy))
        print(f"  exit {code}, outputs {kv}")
        assert (code, kv.get("verdict")) == (0, "PASS"), (code, kv, done.stderr[-400:])
        assert kv.get("overridden") == "false", kv

        print("case 2: BLOCK fails the job")
        code, kv, _, _ = run_action(work, "block", manifest_for(degraded))
        print(f"  exit {code}, outputs {kv}")
        assert (code, kv.get("verdict")) == (1, "BLOCK"), (code, kv)

        print("case 3: INCONCLUSIVE fails the job by default, summary names the hold")
        code, kv, summary, _ = run_action(work, "hold", manifest_for(thin))
        print(f"  exit {code}, outputs {kv}")
        assert (code, kv.get("verdict")) == (1, "INCONCLUSIVE"), (code, kv)
        assert kv.get("overridden") == "false", kv
        assert "held by parse_survival" in summary, summary

        print("case 4: allow-inconclusive passes the job, keeps the verdict, says so")
        code, kv, summary, _ = run_action(work, "override", manifest_for(thin), "true")
        print(f"  exit {code}, outputs {kv}")
        assert code == 0, code
        assert kv.get("verdict") == "INCONCLUSIVE", kv
        assert kv.get("overridden") == "true", kv
        assert "override" in summary.lower(), summary
        assert "INCONCLUSIVE" in summary, summary

        print("case 5: allow-inconclusive does not touch a BLOCK")
        code, kv, _, _ = run_action(
            work, "override-block", manifest_for(degraded), "true"
        )
        print(f"  exit {code}, outputs {kv}")
        assert (code, kv.get("verdict")) == (1, "BLOCK"), (code, kv)
        assert kv.get("overridden") == "false", kv

        print("case 6: a crashed measurement is ERROR and fails the job")
        code, kv, summary, _ = run_action(work, "error", manifest_for(crashing))
        print(f"  exit {code}, outputs {kv}")
        assert (code, kv.get("verdict")) == (1, "ERROR"), (code, kv)
        assert "measurement command failed" in summary, summary

        print("case 7: no report at all is ERROR, not a decision")
        code, kv, _, _ = run_action(
            work, "noreport", manifest_for(healthy, axis="vibes")
        )
        print(f"  exit {code}, outputs {kv}")
        assert (code, kv.get("verdict")) == (1, "ERROR"), (code, kv)

        # The 2026-09-05 review's reproduction: a valid qualification leaves a
        # PASS report in the output directory; a second invocation into the
        # same directory with a manifest the qualifier rejects before it can
        # clean up must not read that old PASS as this run's result.
        print("case 8: valid PASS, then a malformed manifest into the same directory")
        code, kv, _, _ = run_action(work, "reuse", manifest_for(healthy))
        assert (code, kv.get("verdict")) == (0, "PASS"), (code, kv)
        code, kv, summary, done = run_action(work, "reuse", {"candidates": "invalid"})
        print(f"  second invocation: exit {code}, outputs {kv}")
        assert code == 1, (code, done.stderr[-400:], done.stdout[-400:])
        assert kv.get("verdict") == "ERROR", kv
        assert kv.get("overridden") == "false", kv

        print("case 9: valid PASS, then a manifest file that does not exist")
        code, kv, _, _ = run_action(work, "reuse2", manifest_for(healthy))
        assert (code, kv.get("verdict")) == (0, "PASS"), (code, kv)
        (work / "reuse2.json").unlink()
        code, kv, summary, done = run_action(work, "reuse2", None)
        print(f"  second invocation: exit {code}, outputs {kv}")
        assert code == 1, (code, done.stderr[-400:])
        assert kv.get("verdict") == "ERROR", kv

        # Round-2 finding 2: the manifest's session policy must reach every
        # adapter and the report, or be rejected where it cannot be enforced.
        print("case 10: policy 0.1 at manifest level, a one-session command axis")
        code, kv, summary, _ = run_action(
            work,
            "policy1",
            manifest_for(healthy, session_failure_margin=0.1, repeats=1),
        )
        print(f"  exit {code}, outputs {kv}")
        assert (code, kv.get("verdict")) == (1, "INCONCLUSIVE"), (code, kv)
        assert kv.get("overridden") == "false", kv
        assert "held by parse_survival" in summary, summary

        print("case 11: no session policy in the manifest at all")
        code, kv, _, done = run_action(
            work, "nopolicy", manifest_for(healthy, session_failure_margin=None)
        )
        print(f"  exit {code}, outputs {kv}")
        assert (code, kv.get("verdict")) == (1, "ERROR"), (code, kv)
        assert "session-failure margin" in done.stderr + done.stdout, done.stderr[-400:]

        print("case 12: diagnostic manifest reports, never approves")
        code, kv, summary, _ = run_action(
            work, "diag", manifest_for(healthy, diagnostic=True, repeats=29)
        )
        print(f"  exit {code}, outputs {kv}")
        assert (code, kv.get("verdict")) == (1, "INCONCLUSIVE"), (code, kv)
        assert kv.get("scope") == "observed-sessions-diagnostic", kv
        assert "diagnostic" in summary, summary

        print("case 13: manifest policy 0.1, axis override 0.5 with five sessions")
        code, kv, _, _ = run_action(
            work,
            "override-axis",
            manifest_for(
                healthy, session_failure_margin=0.1, repeats=5, axis_policy=0.5
            ),
        )
        print(f"  exit {code}, outputs {kv}")
        assert (code, kv.get("verdict")) == (0, "PASS"), (code, kv)
        report = json.loads(
            (work / "override-axis" / "release_verdict.json").read_text()
        )
        assert report["axes"][0]["policy"]["session_failure_margin"] == 0.5, report[
            "axes"
        ][0]
        assert report["policy"]["session_failure_margin"] == 0.1, report["policy"]

        print("case 14: policy 0.1 with 29 sessions of a healthy command")
        code, kv, _, _ = run_action(
            work,
            "policy29",
            manifest_for(healthy, session_failure_margin=0.1, repeats=29),
        )
        print(f"  exit {code}, outputs {kv}")
        assert (code, kv.get("verdict")) == (0, "PASS"), (code, kv)

    print("\nci action: all cases behaved as planted")


if __name__ == "__main__":
    main()
