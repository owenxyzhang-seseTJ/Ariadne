"""Stable error classes for MCP callers.

An agent has to tell three very different situations apart: fix the arguments and retry,
wait for the same call, or stop and get a human scientific decision. The message prefix
is the contract; the code list is closed so callers can branch without parsing prose.
"""
from __future__ import annotations

CODES = {
    'E_INPUT': 'Arguments are malformed or not applicable; correct them and retry.',
    'E_NOT_FOUND': 'The job, artifact or operation does not exist.',
    'E_BUSY': 'Another worker or mutation holds this job; retry after it finishes.',
    'E_NEEDS_EVIDENCE': 'A preceding stage is not scientifically accepted; no retry will help.',
    'E_IMMUTABLE': 'The target is already accepted or finalized; create a new branch job.',
    'E_BUDGET': 'The attempt or time budget is exhausted; review the cause first.',
    'E_UPSTREAM': 'A child process or workflow script failed; read its log.',
    'E_CANCELLED': 'The operation was cancelled; outputs are not usable as a result.',
}


class ToolError(ValueError):
    """ValueError carrying a stable E_* code, so existing handlers keep working."""

    def __init__(self, code: str, message: str):
        if code not in CODES:
            raise ValueError('Unknown error code: ' + code)
        self.code = code
        super().__init__(f'{code}: {message}')


def error_code(exc) -> str | None:
    """Extract the code from a ToolError or a message that carries one."""
    code = getattr(exc, 'code', None)
    if code in CODES:
        return code
    text = str(exc)
    for candidate in CODES:
        if candidate in text:
            return candidate
    return None
