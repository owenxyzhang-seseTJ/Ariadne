# License scope and third-party notices

Ariadne's independently authored software and documentation are proprietary. Commercial use requires a written license as described in [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md). Releases up to commit 407d193 remain available under the MIT License reproduced in [LICENSE-MIT](LICENSE-MIT).

DIALS-derived source portions retain the upstream BSD license reproduced in [patches/DIALS-LICENSE](patches/DIALS-LICENSE). The commercial license does not replace those upstream conditions. The source patch must be distributed with the corresponding DIALS notice.

Third-party dependencies and referenced publications retain their respective licenses. Official DAFi binaries and proprietary vendor programs are not distributed with this software.

## Bundled third-party components

The runtime installed by `scripts/install_local.sh` includes the components below. Their license texts are reproduced under [`licenses/`](licenses/).

| Component | License | Copyright |
|---|---|---|
| DIALS | BSD-3-Clause | Diamond Light Source, Lawrence Berkeley National Laboratory, United Kingdom Research and Innovation, contributors |
| dxtbx | BSD-3-Clause | Diamond Light Source, Lawrence Berkeley National Laboratory, Science and Technology Facilities Council |
| cctbx | cctbx license (BSD-style) | The Regents of the University of California |
| NumPy | BSD-3-Clause | NumPy developers |
| SciPy | BSD-3-Clause | SciPy developers |

These licenses permit commercial distribution and modification provided the copyright notices, conditions and disclaimers are retained and no endorsement is implied. The Ariadne patch modifies DIALS source and remains subject to the DIALS license; [`patches/DIALS-LICENSE`](patches/DIALS-LICENSE) reproduces that notice alongside the patch.
