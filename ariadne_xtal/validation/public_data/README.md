# Published FeN4 measured-data fixture

The two original files `2242624.cif` and `2242624.hkl` are retained byte for byte,
including their original attribution and reuse headers. They were downloaded
from Gemmi's public crystallographic test fixtures on 2026-10-30. The numerical
source is Crystallography Open Database entry 2242624, originating from the
IUCr supporting data for the article below.

**Required attribution:** Maxim Bykov, Saiana Khandarkhaeva, Timofey Fedotenko,
Pavel Sedmak, Natalia Dubrovinskaia and Leonid Dubrovinsky (2018),
“Synthesis of FeN4 at 180 GPa and its crystal structure from a submicron-sized
grain”, *Acta Crystallographica Section E* **74**, 1392–1395.
DOI: https://doi.org/10.1107/S2056989018012161

Article and original supporting data:

- https://journals.iucr.org/e/issues/2018/10/00/wm5459/index.html
- https://doi.org/10.1107/S2056989018012161/wm5459sup1.cif
- https://doi.org/10.1107/S2056989018012161/wm5459Isup2.hkl

Downloaded copies:

- https://raw.githubusercontent.com/project-gemmi/gemmi/master/tests/2242624.cif
- https://raw.githubusercontent.com/project-gemmi/gemmi/master/tests/2242624.hkl

The original article states the Creative Commons Attribution (CC-BY) license.
The COD fixture headers permit use within the scientific community with
proper attribution to the journal article. The original terms apply to these
third-party data; the project's software license does not replace them.
The result JSON records exact SHA-256 hashes of both downloaded files.

`2242624.hkl` is a **structure-factor CIF**, not a SHELX fixed-column HKLF4
file. `validation/run_real_fen4.py` uses Gemmi's CIF parser to extract only the
71 measured F² values and their uncertainties. The archived Fcalc column is
never used as an observation. The comparison begins with the deposited model
and reports local re-refinement, not ab-initio structure solution.
