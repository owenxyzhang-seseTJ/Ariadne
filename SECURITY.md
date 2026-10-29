# Security policy

## Supported versions

Security fixes are applied to the `main` branch and the most recent release.

## Reporting a vulnerability

Use GitHub private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**. Do not open a public issue for an unpatched vulnerability, and do not attach confidential diffraction data, credentials or customer files. A minimal synthetic reproduction is sufficient.

We aim to acknowledge a report within three business days, provide an initial assessment within ten business days, and coordinate disclosure once a fix is available.

## Scope

Reports of interest include:

- code execution, path traversal or unintended file access through reflection tables, experiment files, patches or job artifacts;
- credential or token leakage through logs, MCP tool output or job state;
- agent tool misuse caused by untrusted input, including prompt injection that escapes the declared operation scope;
- unintended transmission of diffraction data to external services.

## Deployment guidance

Run the numerical workflow locally or in a controlled environment. External model providers must be configured explicitly; keep customer data out of prompts unless a data-processing agreement covers it. Job artifacts and logs can contain file paths and reflection metadata and should be protected accordingly.
