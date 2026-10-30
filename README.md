# Ariadne 0.7.0

A crystallography workflow with numerical reduction tools, persistent agent tasks, native CIF inspection and structure-assisted intensity inference.

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [License](LICENSE)

This release builds on the recovered R2 source and reconstructs the 0.7.0 agent and feedback layers. The repository includes the complete source, DIALS patch, regression tests, examples, Python distribution and technical reports. The [publication review](CODE_REVIEW.zh-CN.md) records the additional corrections made before pushing this version.

## Start here

- [Chinese quick start and operation guide](AGENT_GUIDE.zh-CN.md)
- [Technical report](reports/Ariadne_0_7_0_Technical_Report.pdf) and [LaTeX source](reports/Ariadne_0_7_0_Technical_Report.tex)
- [Actual validation record](reports/rebuild_validation.json)
- [Synthetic native-Gemmi example](examples/agent/create_demo.py)
- [Implementation status](implementation-status.json)
- [Current source and installation checks](reports/github_review_validation.json)

## What is present

The complete R2 numerical code and DIALS Python-source patch are retained. New code supplies real Gemmi CIF auditing and X-ray structure factors, block-local/low-rank Gaussian structural feedback, training-only scale calibration, closed data partitions and final-test exposure records. A persistent agent runs typed scientific actions, resumes completed work, discovers actual paired SHELXT candidates, and records the distinction between execution and scientific acceptance.

The agent supports audit, solve, refine, reduce, feedback and explicit custom DAGs. It has 13 registered action types, 8 additional optional MCP session tools, a deterministic policy, and an optional Chat Completions JSON planner. External executables are configured paths, never model-authored shell commands.

## Minimal demonstration

Use Python 3.12 or newer with NumPy, SciPy and Gemmi. The demo does not need DIALS, SHELX, a language model or MCP.

```bash
export PYTHONPATH="$PWD/python/src${PYTHONPATH:+:$PYTHONPATH}"
python -m ariadne.agent.cli capabilities
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/agent/create_demo.py /absolute/path/to/NEW-DEMO
```

To install CLI entry points and the MCP dependency, use `python -m pip install ./python`. The wheel in `dist/` contains the Python package, not third-party runtimes. Keep this source directory for the original analysis scripts and expert packs; set `ARIADNE_PLUGIN_ROOT` when needed. Linux, macOS and WSL are supported by the agent's POSIX execution lock.

## Validation scope

The publication review ran **323 passing tests, 11 skips and no failures**, plus 136 passing subtests. The skips require DIALS/dxtbx or cctbx. Real Gemmi computations and real MCP registration were executed. The built wheel was installed into a separate directory and completed a three-action CIF audit from outside the source checkout. See [current validation](reports/github_review_validation.json).

SHELXT/SHELXL and model-service integration use explicitly labelled protocol fixtures in this environment. Experimental raw-frame accuracy remains unverified. The PDF's earlier validation totals describe the reconstruction run; the publication review is the current record.

The automatic raw-image route is single-crystal. Complex twins require explicit joint-analysis workflows. Structural feedback is conditional on a fixed Gaussian observation model; the real-IT92 predictor does not include anomalous scattering or bulk-solvent corrections. Signed observations and model-assisted posteriors are different objects.

## Verify and reproduce

```bash
python scripts/verify_release.py
python -m pip install -e './python[dev]'
python scripts/run_review_validation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/feedback_rebuild.py
```

The release manifest verifies delivered bytes, not scientific truth. Re-running tests updates logs, so subsequent manifest differences in those logs are expected.

The original scientific installer, `sh scripts/install_local.sh`, creates a source overlay for the matching DIALS **3.30.1** conda runtime. Use a fresh runtime directory. A standalone pip cctbx installation is not a replacement for that matched native environment.

## Historical material and licensing

Historical R2 guides are preserved under `docs/R2_README*`. Previous reports, benchmarks and patches remain available, but their native-runtime performance and test claims were not reproduced here unless explicitly recorded in `reports/rebuild_validation.json`. The original DIALS target remains 3.30.1; use the supplied installer only in a fresh matching environment.

Existing LICENSE, LICENSING.md and dependency notices apply. SHELXT/SHELXL, DIALS compiled binaries and fonts are not redistributed in this package.
