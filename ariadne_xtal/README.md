# Ariadne-Xtal

**面向科研 agent 的单晶结构求解、精修与绘图内核。**

Ariadne-Xtal 把衍射计算组织为可以直接调用、检查和复现的数值操作：读取反射强度，提出结构候选，精修明确的模型，分析残差与不确定度，输出结构、密度图和交互图形。每次 agent 任务保存输入快照、参数、环境版本、结果和文件哈希。语言模型负责提出任务，数值内核负责计算。

本源码包包含独立实现的原生算法，以及连接用户已有 SHELXT/SHELXL 的显式执行接口。**0.1.0 不是全部 SHELXT/SHELXL 功能的独立等价重写。** 原生模块已实现并测试的范围见下表；保留完整原始指令语义时，选择单独安装的官方后端。源码包不包含官方程序。

## 计算范围

| 模块 | 原生实现 |
|---|---|
| 晶体几何 | 一般晶胞、正倒易度量、精确周期最近像搜索 |
| 对称性 | Gemmi 空间群表、完整显式操作验证、特殊位置、候选群消光和等价反射诊断 |
| 散射 | 独立的运动学结构因子求和、中性原子 IT92、各向同性/各向异性 ADP、显式异常散射 |
| 初始求解 | 已知晶胞组成的 P1 多起点 charge flipping 和稀疏原子投影 |
| 精修 | 全部工作集 F²、比例因子、坐标、ADP、可选占有率和完全重叠孪晶比例 |
| 约束与限制 | 位置对称性、连续原点自由度、相等参数、占有率和；七类显式几何/ADP 限制 |
| 不确定度 | 雅可比 SVD、可辨识性、条件协方差、物理参数标准不确定度、按反射轨道分组的留出集 |
| 绝对结构 | 反演孪晶比例精修；固定结构的加权 Friedel quotient 诊断 |
| 密度图 | 非加权 Fo−Fc、2Fo−Fc、计算密度、Patterson、周期峰搜索、CCP4/NPZ |
| 绘图 | 离线三维位移椭球和密度等值面、倒易空间、残差及相关矩阵、SVG/PDF |
| Agent | Python、CLI、JSONL、可选 MCP；显式后端选择、独立运行目录、输入输出追踪 |
| 结构反馈 | 带交叉拟合来源声明和信息强度上限的高斯轮廓解混研究接口 |

P1 求解模块要求提供**整个晶胞**的元素列表。它不实施 SHELXT 完整的 Laue 群、相位空间群判定、原点搜索与元素指认流程。原生精修也不执行 AFIX/HFIX、完整自由变量语言、HKLF5 非完全重叠孪晶、中子精修、EXTI、ABIN/SWAT 或完整 CIF 归档规则。这些工作通过原始指令文件交给明确启用的官方后端。

完整边界可直接查询：

```bash
ariadne-xtal capabilities
```

SHELXL 公开索引中 77 条指令的逐项状态见 [兼容性说明](docs/COMPATIBILITY.md) 和 [机器可读矩阵](shelx_instruction_matrix.json)。

## 安装

基础包声明支持 Python 3.10 及以上；本交付在 Python 3.12 环境运行验证。建议为项目创建独立虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境，后续 pip 命令相同。安装会从包索引获取 NumPy、SciPy、Gemmi、Matplotlib、Plotly 和 scikit-image。运行已经生成的 HTML 图形不需要网络。

只使用命令行和 Python 时，无需安装 MCP。需要接入支持 MCP 的 agent 客户端时：

```bash
python -m pip install -e '.[mcp]'
ariadne-xtal-mcp --root /absolute/path/to/project
```

MCP 传输使用官方 Python SDK。本交付实际验证 SDK 2.3.0，代码同时保留 v1 FastMCP 导入兼容路径。建议 MCP 环境使用 Python 3.11 或更高版本；具体最低版本以所安装 SDK 的元数据为准，当前已验证的 2.3.0 元数据声明为 Python ≥3.10。

## 运行第一个精修任务

在源码目录执行：

```bash
python examples/generate_examples.py
ariadne-xtal run examples/refine_job.json --root .
```

等价的直接命令为：

```bash
ariadne-xtal refine examples/demo_start.json \
  --hkl examples/demo_reflections.json \
  --root .
```

结构化请求的基本形式：

```json
{
  "action": "refine",
  "structure": "examples/demo_start.json",
  "hkl": "examples/demo_reflections.json",
  "options": {
    "max_nfev": 300,
    "origin_gauge": "auto",
    "covariance_scale": "absolute"
  },
  "graphics": true
}
```

`structure` 和 `hkl` 是 `--root` 下的路径，不能引用工作区外部文件。请求文件本身由 CLI 按普通文件路径读取。输出写入新的 `runs/refine_.../` 目录，响应中给出 `manifest` 和 `result` 的相对路径。

运行结果包括模型 JSON、CIF、可表示时的结构 RES、密度图、离线 HTML 和静态图。`result.json` 保存比例因子、指标、参数化、协方差、秩、约束、限制和诊断信息；`manifest.json` 保存任务及文件来源。

返回的 `ok: true` 表示任务执行完成。精修的数值收敛由 `result.success` 表示，参数是否可辨识由 `result.metrics.identifiable` 表示。二者需要结合结构化学、数据质量、残差和替代模型共同判断。

## Python 使用

```python
from ariadne_xtal.io import read_structure, read_reflections, write_structure
from ariadne_xtal.refinement import RefinementOptions, refine

model = read_structure("examples/demo_start.json")
data = read_reflections("examples/demo_reflections.json")
fit = refine(model, data, RefinementOptions(max_nfev=300))

print(fit.success, fit.metrics["identifiable"])
print(fit.metrics["r1_gt"], fit.metrics["wr2"])
write_structure(fit.structure, "refined.cif")
```

函数不原地修改传入的结构或反射表。原生精修保留负测量强度，目标函数为 F² 加权残差；开方截断只用于需要振幅的初始求解、R1 和诊断密度图。

更多精确约束、几何限制、孪晶、留出集、绘图和反馈示例见 [API 文档](docs/API.md)。

## 为 agent 提供工具

### JSONL

```bash
ariadne-xtal serve --root /absolute/path/to/project
```

标准输入每行一个 JSON 请求，标准输出每行一个 JSON 响应。支持 `capabilities`、`inspect`、`solve`、`refine`、`maps`、`absolute` 和 `legacy`。例如：

```json
{"action":"inspect","structure":"examples/demo_start.json","hkl":"examples/demo_reflections.json"}
```

### MCP

客户端启动配置中的可执行命令为 `ariadne-xtal-mcp`，参数为 `--root` 及项目绝对路径。服务公开两个工具：

- `crystal_capabilities()`：读取当前实现范围与动作列表。
- `crystal_execute(request)`：执行结构化计算请求。

服务没有任意 Python 或 shell 执行工具。外部引擎只能由启动配置明确启用。任务预算限制了输入文件、原子数、反射数和求解规模；这些限制用于控制数值工作量。

### 官方 SHELXT/SHELXL 后端

先把官方程序安装到本机，并将完整 INS/HKL 及附属文件放入同一项目目录树。用以下方式启用明确的可执行文件：

```bash
ariadne-xtal serve --root /absolute/path/to/project \
  --shelxt /absolute/path/to/shelxt \
  --shelxl /absolute/path/to/shelxl
```

然后提交请求：

```json
{
  "action": "legacy",
  "engine": "shelxl",
  "input": "crystal/sample.ins",
  "timeout": 300,
  "extra_args": []
}
```

原始指令与反射字节在独立目录中交给所选程序，不进行原生转换；同名 HKL/FAB、项目内嵌套 include 及明确指定的附属文件会一同复制。各类复杂精修功能由该官方程序的已安装版本实施。原生任务失败不会自动切换到官方后端。

后端返回码为 0 时记录为 `completed`，其含义是进程正常退出。兼容接口同时记录 stdout、stderr、可执行文件与数据哈希；当前交付的传输验证使用 mock executables，没有运行官方程序的数值对照。

## 输入文件与科学约定

原生内部使用 Å、度、分数坐标和**笛卡尔 U 张量**。倒易基矢不含 2π；占有率是物理位点占有率，完整占据的特殊位置仍为 1。SHELX 特殊位置 SOF 在导入和导出时按重数换算。

JSON 是保留全部原生字段和浮点精度的主要格式。CIF 提供结构互操作；完整显式对称操作优先于 H–M 名称。HKLF4 保留负强度和 batch，不会自动合并反射或删除弱反射。负 batch 只有在明确要求时才解释为留出标记。

INS/RES 分为两条处理路径：`ShelxDocument` 原样保存文档；`to_structure()` 只转换有明确原生含义的结构子集。遇到未翻译指令、自由变量关系或非零晶胞不确定度时停止转换，返回具体位置。CIF 结构导入不重建嵌入的 SHELXL 精修配方。

异常散射项由 `Atom.f_prime` 和 `Atom.f_double_prime` 明确给出。可通过 `with_anomalous_factors()` 主动调用 Gemmi 的 Cromer–Liberman 计算。它使用孤立原子模型，不代替吸收边附近的实测光谱校准。

## 实测数据回归

源码包保留公开 FeN₄ 数据（COD 2242624）及复现脚本：

```bash
python validation/run_real_fen4.py
```

该实验使用已发表的 71 个测量 F² 和对应标准不确定度，从存档坐标出发精修 10 个参数。脚本只读取测量列，不把存档 Fcalc 当作观测；采用存档中更高精度的晶胞及已给定权重参数。

| 指标 | 原生重精修 | 论文存档显示值 |
|---|---:|---:|
| R1，I > 2σ(I) | 0.04031575 | 0.0403 |
| R1，全部反射 | 0.04126822 | 0.0413 |
| wR2 | 0.08148047 | 0.0815 |
| GoF | 1.18332441 | 1.183 |

对 N 坐标施加最高 0.02 的分数坐标扰动后，第二个初值收敛到相同局部解。这是一个已知结构的实测数据回归，检验前向模型与局部精修的一致性；它不测量从头求解成功率、运行速度优势或完整 SHELXL 数值等价性。

数据来源为 Bykov 等的 [FeN₄ 原始论文及支持数据](https://doi.org/10.1107/S2056989018012161)。原始文件、许可和署名保留在 [public_data](validation/public_data/README.md)；完整数值、输入哈希及实验设定见 [回归结果](validation/real_fen4/result.json)。

## 验证与技术报告

```bash
python -m pytest -q
python -m ruff check src tests
```

测试包括几何与倒易度量不变量、Gemmi 结构因子对照、对称和特殊位置、孪晶强度、模拟数据恢复、约束/限制、秩亏与留出隔离、密度图、输入格式和 mock 后端传输。单项测试记录在 `tests/`；实测回归单独记录在 `validation/`。

[技术报告 PDF](docs/report.pdf)从散射、傅里叶表示、空间群和相位问题出发，推导精修、约束、协方差、孪晶、交叉拟合反馈及位移椭球。报告中的数学模型与对应代码模块相互索引。[LaTeX 主文件](docs/report.tex)、章节和矢量插图随源码提供，重新编译方法见 [BUILD.md](docs/BUILD.md)。

无需运行计算即可打开随包提供的 [离线绘图示例](examples/demo_output/dashboard.html)。示例目录同时保留模型、密度图、静态图和对应的任务记录。

## 代码组织

| 路径 | 内容 |
|---|---|
| `src/ariadne_xtal/model.py`、`symmetry.py` | 数据类型、晶胞与对称 |
| `scattering.py`、`phasing.py` | 散射与 P1 候选求解 |
| `refinement.py`、`restraints.py` | 参数化、精修和几何/ADP 限制 |
| `maps.py`、`visualization.py` | 傅里叶图、三维和静态绘图 |
| `absolute.py`、`feedback.py` | 异常散射、绝对结构诊断、轮廓反馈 |
| `io.py`、`shelx.py`、`compat.py` | 文件与明确的官方后端执行 |
| `agent.py`、`cli.py`、`mcp_server.py` | 任务接口与来源记录 |
| `docs/`、`tests/`、`validation/` | 技术文档、测试与回归证据 |

本项目源码采用 [MIT 许可证](LICENSE)。依赖软件和第三方晶体学数据保留各自的许可与署名。SHELXT、SHELXL 名称用于标识独立安装的外部程序；本项目不是其官方发行版。
