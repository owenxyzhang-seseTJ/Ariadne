# Reports and validation records

## Current reconstructed build

- `github_review_validation.json`: current publication checks, including the complete collected suite, package build and installed-wheel audit.
- [Source review and corrections](../CODE_REVIEW.zh-CN.md): issues found before publication and their regression coverage.

- `Ariadne_0_7_0_Technical_Report.pdf` / `.tex`: current textbook-style Chinese report.
- `rebuild_validation.json`: reconstruction-suite record; see current publication checks for the broader suite.
- `rebuild_feedback_benchmark.json`: newly measured fixed-Gaussian feedback benchmark.
- `rebuild_demo.json`: actual native-Gemmi synthetic feedback and resume demonstration.
- `rebuild/`: detailed logs and JUnit output for this run.

## Historical records

All other reports and measurements originated in the recovered R2 archive unless explicitly named as a reconstruction record. Earlier native DIALS executions and performance claims are historical; the current container did not repeat those native-runtime experiments. `inherited-delivery.json` and `inherited-implementation-status.json` preserve the previous metadata.

No report in this release demonstrates real experimental MOF structure-solution accuracy. External SHELXT/SHELXL are tested using protocol fixtures; Gemmi computations are real native calls.
