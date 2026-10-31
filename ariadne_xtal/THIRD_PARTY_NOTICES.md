# Third-party notices and provenance

The Ariadne-Xtal Python implementation is independently written and licensed under the accompanying MIT license. No official SHELXT or SHELXL source or executable is distributed. Public algorithm descriptions and file-format definitions are cited in the report and compatibility documentation.

Runtime dependencies are installed separately and retain their upstream licenses:

| Dependency | Role | Upstream |
|---|---|---|
| NumPy | Arrays and Fourier transforms | https://numpy.org/ |
| SciPy | Optimization, matrix and statistical functions | https://scipy.org/ |
| Gemmi | Published scattering-factor tables, space-group tables, CIF and CCP4 I/O | https://github.com/project-gemmi/gemmi |
| Matplotlib | Static scientific figures | https://matplotlib.org/ |
| Plotly | Interactive figures and the JavaScript runtime embedded in generated HTML | https://github.com/plotly/plotly.js |
| scikit-image | Marching-cubes surface extraction | https://scikit-image.org/ |
| Official MCP Python SDK (optional) | Standard agent transport | https://github.com/modelcontextprotocol/python-sdk |

The embedded Plotly JavaScript retains its upstream copyright and license header. Its MIT license text is reproduced below. Generated figures and document text are original to this project except where an explicit attribution is given.

The real FeN4 validation fixture is original IUCr/COD scientific data mirrored in Gemmi's tests. Its exact download locations, original article, file hashes and reuse/attribution notices are retained in `validation/public_data/README.md` and in the original file headers. It is not synthetic data.

The optional report build helper downloads Noto Serif CJK fonts from the official Noto repository. Font files are not included in this source archive; the helper downloads their SIL Open Font License alongside them.

## Plotly.js MIT license

Copyright (c) 2012-2026 Plotly, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
