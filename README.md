# InferenceGate: synthetic demonstration candidate

Experimental evidence gate for paired inference-serving comparisons. PASS, BLOCK,
INCONCLUSIVE, and ERROR are distinct outcomes. This isolated extraction has no
historical model records, paper, private Git history, or vLLM parser adapter.

Python 3.11+; runtime uses the standard library. Run `bash DEMO.sh` for planted
mock endpoint scenarios, or `bash run_gate_tests.sh` for the synthetic suite.
Only localhost networking is used by the suite. Mock latency and quality are
simulated; these checks are not measurements of a real model or production system.
No certification or universal false-PASS guarantee is claimed.

Supported scope is the normal replay/qualification path exercised by the suite.
The inherited replay CLI also exposes legacy `--retry-invalid`/`--retry-feedback`
research options. They require excluded repair-analysis modules and are not
supported by this minimal package. The example workflow is a template requiring
your own trusted serving stacks and release manifest, not a runnable hosted demo.

The composite action fails closed unless its explicitly documented inconclusive
override is enabled. Command manifests intentionally execute shell commands:
use only trusted manifests. Environment quoting prevents wrapper input injection;
it does not sandbox commands or untrusted endpoint responses.

Source provenance: personal project, sole human creator David Wu (confirmed
2026-09-30); AI-assisted engineering. Retained original files are from private
source commit 281432ab5efa0bbbc3cea050a8b55d680cdcd72e. Licensed under the MIT License; see LICENSE.
See THIRD_PARTY_DEPENDENCIES.md for external components and retained terms.
