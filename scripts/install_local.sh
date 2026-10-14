#!/bin/sh
# Install Python-source modifications over the matching compiled DIALS runtime.
# No benchmarks, smoke tests, or algorithm invocations are performed here.
set -eu
RF_PLUGIN_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RF_BASE_PREFIX=${1:-"$HOME/CrystallographyPrograms/scxrd-open"}
RF_RUNTIME=${2:-"$HOME/CrystallographyPrograms/ariadne"}
RF_PROGRAMS="$HOME/CrystallographyPrograms"
if [ ! -x "$RF_BASE_PREFIX/bin/python" ]; then
    RF_MAMBA=${ARIADNE_MAMBA:-"$HOME/miniforge3/bin/mamba"}
    if [ ! -x "$RF_MAMBA" ]; then
        printf '%s\n' "Set ARIADNE_MAMBA to a working mamba executable." >&2
        exit 2
    fi
    "$RF_MAMBA" create -p "$RF_BASE_PREFIX" -c conda-forge --override-channels \
        python=3.12 dials=3.30.1 dxtbx=3.30.0 fabio gemmi numpy scipy matplotlib-base pyyaml -y
fi
# Read installation metadata only; this is not a DIALS runtime test.
RF_DIALS_RECORD=$(find "$RF_BASE_PREFIX/conda-meta" -maxdepth 1 -name 'dials-3.30.1-*.json' -print | head -n 1)
if [ -z "$RF_DIALS_RECORD" ]; then
    printf '%s\n' "Expected a conda DIALS 3.30.1 runtime at $RF_BASE_PREFIX." >&2
    exit 2
fi
mkdir -p "$RF_RUNTIME/python" "$RF_RUNTIME/downloads" "$RF_PROGRAMS"
RF_ARCHIVE=${ARIADNE_SOURCE_ARCHIVE:-"$RF_RUNTIME/downloads/dials-v3.30.1.tar.gz"}
if [ ! -f "$RF_ARCHIVE" ]; then
    curl -fL --retry 2 https://codeload.github.com/dials/dials/tar.gz/refs/tags/v3.30.1 -o "$RF_ARCHIVE"
fi
RF_SOURCE="$RF_RUNTIME/dials-3.30.1"
if [ -e "$RF_SOURCE" ]; then
    printf '%s\n' "Source already exists: use a fresh runtime directory for a new build." >&2
    exit 2
fi
tar -xzf "$RF_ARCHIVE" -C "$RF_RUNTIME"
patch -d "$RF_SOURCE" -p1 < "$RF_PLUGIN_ROOT/patches/dials-3.30.1-ariadne.patch"
if [ ! -d "$RF_RUNTIME/python/mcp-1.30.0.dist-info" ]; then
    "$RF_BASE_PREFIX/bin/python" -m pip install --target "$RF_RUNTIME/python" 'mcp==1.30.0'
fi
"$RF_BASE_PREFIX/bin/python" -m pip install --no-deps --target "$RF_RUNTIME/python" "$RF_PLUGIN_ROOT/python"
"$RF_BASE_PREFIX/bin/python" - "$RF_PLUGIN_ROOT" "$RF_BASE_PREFIX" "$RF_RUNTIME" "$RF_PROGRAMS" <<'PY'
import json
from pathlib import Path
import shlex
import sys
plugin, base, runtime, programs = map(Path, sys.argv[1:])
record = {
    'schema': 'ariadne-runtime/1', 'product': 'Ariadne', 'version': '0.5.0',
    'base_prefix': str(base.resolve()), 'source': str((runtime / 'dials-3.30.1').resolve()),
    'compiled_extensions': 'unmodified DIALS 3.30.1/cctbx/dxtbx conda binaries',
    'deployment': 'patched DIALS Python source plus companion package on explicit PYTHONPATH',
    'mcp_sdk': '1.30.0', 'plugin_root': str(plugin.resolve()),
    'baseline_launcher': str(programs / 'scxrd-env'), 'tests': 'not_run_as_requested',
    'algorithm_validation': 'not_run', 'format_compatibility': 'actual Bruker/Rigaku datasets unverified',
    'global_composite_absorption_estimator': 'not_implemented'
}
(runtime / 'runtime.json').write_text(json.dumps(record, indent=2) + '\n')
# Local desktop clients do not necessarily inherit a login-shell PATH.
config = {'mcpServers': {'ariadne': {
    'command': str(programs / 'ariadne-env'),
    'args': ['python', '-m', 'ariadne.server']}}}
(plugin / '.mcp.json').write_text(json.dumps(config, indent=2) + '\n')
values = {
    'ARIADNE_PLUGIN_ROOT': str(plugin.resolve()),
    'SCXRD_RUNTIME_CONFIG': str((runtime / 'runtime.json').resolve()),
    'ARIADNE_SOURCE': str((runtime / 'dials-3.30.1').resolve()),
}
lines = ['#!/bin/sh'] + ['export ' + key + '=' + shlex.quote(value) for key, value in values.items()]
lines += [
    'export PYTHONNOUSERSITE=1',
    'export PYTHONPATH=' + shlex.quote(str(runtime / 'dials-3.30.1/src') + ':' + str(runtime / 'python')),
    'export PATH=' + shlex.quote(str(base / 'bin') + ':' + str(programs)) + ':"$PATH"',
    'if [ "$#" -eq 0 ]; then',
    '  printf "%s\\n" "Usage: ariadne-env COMMAND [ARGS...]"',
    '  exit 0',
    'fi',
    'exec "$@"',
]
launcher = programs / 'ariadne-env'
launcher.write_text('\n'.join(lines) + '\n')
launcher.chmod(0o755)
print('Installed Ariadne source overlay and launcher:', launcher)
print('Scientific algorithms and MCP runtime were not executed.')
PY
