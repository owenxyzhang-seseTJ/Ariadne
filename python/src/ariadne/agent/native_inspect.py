"""Run with the configured scientific Python, not a stand-in EXPT parser."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def inspect(expt_path, refl_path=None):
    from dxtbx.model.experiment_list import ExperimentListFactory
    from dials.array_family import flex
    experiments = ExperimentListFactory.from_json_file(str(expt_path), check_format=False)
    crystals = {id(e.crystal) for e in experiments if e.crystal is not None}
    result = {'n_experiments': len(experiments), 'n_crystals': len(crystals), 'native_backend': 'dxtbx/dials'}
    if refl_path:
        refs = flex.reflection_table.from_file(str(refl_path))
        result['n_reflections'] = len(refs)
        if 'id' in refs and len(refs):
            result['assigned_fraction'] = float((refs['id'] >= 0).count(True)/len(refs))
    return result


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) not in (2, 3):
        raise SystemExit('usage: native_inspect EXPT [REFL] OUTPUT_JSON')
    result = inspect(args[0], args[1] if len(args) == 3 else None)
    Path(args[-1]).write_text(json.dumps(result, allow_nan=False)+'\n')
