# API 与数据约定

本文对应 Ariadne-Xtal 0.1.0。接口以 `src/ariadne_xtal/` 中的代码为准；[兼容性说明](COMPATIBILITY.md)单独说明原生模型与 SHELXL 指令的关系。

## 1. 基础类型

```python
from ariadne_xtal import Atom, Cell, Reflections, Structure
```

### `Cell`

```python
cell = Cell(a=8.0, b=9.0, c=10.0, alpha=90, beta=103, gamma=90)
```

长度单位为 Å，角度为度。`cell.matrix` 为以晶胞基矢作列的矩阵 (A)，`cell.reciprocal` 为 (A^{-\mathsf T})，不含 (2\pi)。

| 接口 | 含义 |
|---|---|
| `matrix`, `reciprocal`, `metric`, `volume` | 正基、倒基、度量及体积 |
| `to_cartesian(frac)` | 末轴长度为 3 的分数坐标转笛卡尔坐标 |
| `to_fractional(cart)` | 逆变换 |
| `stol_sq(hkl)` | ((\sin\theta/\lambda)^2=\lVert A^{-\mathsf T}h\rVert^2/4) |
| `d_spacing(hkl)` | 晶面间距；000 对应无穷大 |
| `minimum_image(delta_frac)` | 精确最近周期像的分数位移；适用于非正交晶胞 |
| `to_dict()`, `Cell.from_dict(data)` | 数据序列化 |

### `Atom`

```python
atom = Atom(
    label="C1", element="C", frac=[0.12, 0.23, 0.34],
    occupancy=1.0, u_iso=0.02, u_aniso=None,
    f_prime=0.0, f_double_prime=0.0,
    refine_xyz=True, refine_u=True, refine_occupancy=False,
)
```

`occupancy` 为 0–1 的物理占有率，不采用 SHELX 的特殊位置重数因子。`u_iso` 单位为 Å²；`u_aniso` 是对称、半正定的 **3×3 笛卡尔 U 张量**。有 `u_aniso` 时，散射由张量计算。异常散射项由调用者明确给出，默认均为零。

`refine_xyz`、`refine_u` 控制整组坐标或 ADP 是否自由；本基础类型不使用三个彼此独立的坐标固定标志。特殊位置自由度由精修器从初始位点对称性构造。占有率默认固定；主动启用后仍需保证比例因子与占有率之间存在可辨识的标尺。

### `Structure`

```python
structure = Structure(
    cell=cell, atoms=[atom], spacegroup="P 1",
    wavelength=0.71073, sym_ops=None,
)
```

`sym_ops=None` 时使用所声明空间群设定；显式 `sym_ops` 必须包含完整群，包括单位操作与中心化操作，并通过封闭性及晶胞度量检查。显式操作优先于群名。`atoms` 是独立位点列表，标签不区分大小写地要求唯一。`operations` 返回 Gemmi 操作列表。

结构 JSON 示例：

```json
{
  "schema_version": 1,
  "cell": {"a": 8, "b": 9, "c": 10, "alpha": 90, "beta": 103, "gamma": 90},
  "spacegroup": "P 1",
  "wavelength": 0.71073,
  "sym_ops": ["x,y,z"],
  "atoms": [
    {
      "label": "C1", "element": "C", "frac": [0.12, 0.23, 0.34],
      "occupancy": 1, "u_iso": 0.02, "u_aniso": null,
      "f_prime": 0, "f_double_prime": 0,
      "refine_xyz": true, "refine_u": true, "refine_occupancy": false
    }
  ]
}
```

### `Reflections`

```python
data = Reflections(
    hkl=[[1, 0, 0], [0, 1, 0]],
    intensity=[-1.2, 100.0], sigma=[2.0, 5.0],
    batch=None, free_flags=None,
)
```

`hkl` 为 N×3 整数；强度和标准不确定度均为有限数组，`sigma` 严格为正。负测量强度保留。`free_flags=True` 表示留出反射；未提供标记时所有反射用于精修。`subset(mask)` 返回新的反射对象。

反射 JSON 没有顶层 `schema_version`：

```json
{
  "hkl": [[1, 0, 0], [0, 1, 0]],
  "intensity": [-1.2, 100.0],
  "sigma": [2.0, 5.0],
  "batch": null,
  "free_flags": null
}
```

## 2. 文件接口

```python
from ariadne_xtal.io import (
    read_structure, write_structure,
    read_reflections, write_reflections,
    read_hkl, write_hkl,
)
```

| 函数 | 支持范围 |
|---|---|
| `read_structure(path, block_name=None, diagnostics=None)` | JSON、CIF、INS、RES；后两者实施严格原生结构转换 |
| `write_structure(structure, path)` | JSON、CIF 或新的结构 INS/RES，不是原文档编辑 |
| `read_reflections(path)` | `.json` 读取原生反射数据；其他扩展名按 HKLF4 读取 |
| `write_reflections(data, path)` | `.json` 完整数值序列化；其他扩展名写 HKLF4 |
| `read_hkl(path, negative_batch_is_free=False)` | 固定宽度或空白分隔 HKLF4，保留原始 batch |
| `write_hkl(data, path, fixed_width=True)` | 标准定宽输出，或明确指定的空白分隔输出 |

带 `.hkl` 后缀的文件也可能实际是 structure-factor CIF。本通用 HKLF4 读取器不会自动解释该格式。公开 FeN₄ 回归脚本示范了如何用 Gemmi 提取 `_refln_F_squared_meas` 和 `_refln_F_squared_sigma`；不要把存档中的计算强度列当成观测。

CIF 必须提供完整晶胞、明确对称设定或操作、一个波长、中性元素类型、分数坐标及完整位移参数。各向同性 B 可以换算为 U；各向异性 B-only 表、离子散射类型和中子/电子模型不在该读取器范围。多结构块需给出 `block_name`。非零色散项从原子类型表或原生自定义字段读取。

标准 CIF/SHELX 张量与内部张量的关系为：

\[
U_{\rm cart}=A D U_{\rm CIF} D A^{\mathsf T},\qquad
D=\operatorname{diag}(|a^*|,|b^*|,|c^*|).
\]

SHELX 原始文档的保留接口：

```python
from ariadne_xtal.shelx import ShelxDocument

document = ShelxDocument.from_path("sample.ins")
print(document.audit())
document.write("unchanged_copy.ins")  # 原始文本字节保留
model = document.to_structure()      # 未支持语义会抛出 NativeConversionError
```

SHELX 原子 SOF 与物理占有率满足 (o={\rm SOF}\,m_{\rm general}/m_{\rm site})。原生 RES 导出要求可表示的原点设定及 1–4 字符标签；agent 将不能生成的可选 RES 记录为 `artifact_warnings`，JSON/CIF 仍保留数值模型。

## 3. 前向模型与对称诊断

```python
from ariadne_xtal.scattering import structure_factors, calculated_intensities

f = structure_factors(structure, data.hkl)
i_calc = calculated_intensities(structure, data.hkl, scale=1.0)
```

`structure_factors` 返回复结构因子，采用正相位 (\exp(2\pi i h\cdot x))，特殊位置只累加唯一像。等价位点的 U 张量随对称操作旋转；位点稳定子相容性通过张量投影处理。

完全重叠孪晶采用非相干强度和：

```python
i_calc = calculated_intensities(
    structure, data.hkl, scale=2.0,
    twin_laws=[[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]],
    twin_fractions=[0.8, 0.2],
)
```

`twin_laws` **不包括单位变换**，`twin_fractions` **包括单位畴且置于首位**。矩阵作用于倒易列向量 (h'=Th)，数组按 `hkl @ T.T` 计算。矩阵须为保持倒易度量的整数幺模矩阵；不能据此处理一般 HKLF5 不完全重叠组。

```python
from ariadne_xtal.symmetry import systematic_absences, rank_spacegroups

absent = systematic_absences(structure, data.hkl)
diagnostics = rank_spacegroups(cell, data, ["P 1", "P -1"])
```

候选群分数表示已测消光及等价反射的矛盾程度，不是空间群后验概率。消光关系相同的候选群不能靠这个分数唯一判定。

## 4. 原生精修

```python
from ariadne_xtal.refinement import RefinementOptions, refine

fit = refine(structure, data, options=None, restraints=None, constraints=None)
```

`options` 接受 `RefinementOptions` 或同字段字典。输入结构和反射对象不被原地修改。

| 选项 | 默认 | 含义 |
|---|---:|---|
| `max_nfev` | 300 | 每次外层求解的 SciPy 目标评估预算；不包括其差分构造雅可比的额外评估 |
| `ftol`, `xtol`, `gtol` | 1e−9、1e−9、1e−8 | 数值停止容差 |
| `scale` | `None` | 强度比例初值；为空时从工作数据估计 |
| `refine_scale` | `True` | 是否精修强度比例；固定时须提供 `scale` |
| `weight_a`, `weight_b` | 0、0 | 两参数模型相关权重；全零时为标准不确定度权重 |
| `weight_cycles` | 4 | 冻结权重的外层 IRLS 次数 |
| `twin_laws` | `[]` | 非单位倒易孪晶矩阵 |
| `twin_fractions` | `None` | 包含单位畴的比例；为空时均分 |
| `refine_twin_fractions` | `False` | 通过 softmax 精修畴比例 |
| `origin_gauge` | `"auto"` | `auto` 或 `none` |
| `covariance_rcond` | 1e−9 | 雅可比 SVD 的相对截断阈值 |
| `covariance_scale` | `"absolute"` | `absolute` 或 `reduced_chi2` |
| `u_min`, `u_max` | 1e−6、2.0 | Uiso/Cholesky 因子/独立对数模的参数范围 |
| `verbose` | 0 | SciPy 输出等级；agent 强制为 0 |

非零 `weight_a` 或 `weight_b` 时：

\[
P_h=\frac{\max(I_h,0)+2I_h^{\rm calc}}3,\qquad
w_h=\left[\sigma_h^2+(aP_h)^2+bP_h\right]^{-1}.
\]

每个内层求解保持权重固定；结果报告最终实际优化的权重及其与重新计算权重的相对变化。这不是完整六参数 WGHT 指令解释器。

### 精确约束 `constraints`

```json
[
  {"type": "equal_xyz", "atoms": ["C1", "N1"]},
  {"type": "equal_adp", "atoms": ["C1", "N1"]},
  {"type": "occupancy_sum", "atoms": ["C1", "N1"], "target": 1.0}
]
```

| 类型 | 初值与字段要求 |
|---|---|
| `equal_xyz` | 至少两个标签，初始位置模晶格平移相同 |
| `equal_adp` | 初始 ADP 相等，全部为各向同性或全部为各向异性 |
| `equal_occupancy` | 初始占有率相等 |
| `occupancy_sum` | 至少两个正占有率，初始和满足 `target`，所有成员 `refine_occupancy=True` |

`target` 仅用于 `occupancy_sum`，范围为 (0<target\leq1)，默认 1。占有率和组不能与其他占有率约束重叠。相等参数组中有固定成员时，整组相应参数固定。所有占有率与比例因子同时自由、且没有非零固定占有率或精确和约束时，精修器拒绝该不可辨识设定。

位置对称性和原点规范自动加入参数化。`origin_gauge="auto"` 只消除共同的连续极性平移方向；所有位置自由时固定第一个坐标组在这些方向的投影。非极性结构不会因此额外固定一个原子。已经固定坐标时不再添加该自动规范。

位点约束依据**初始**对称性建立；精修器不会自动把一般位置提升为特殊位置。晶胞不参加原生精修。

### 几何与 ADP 限制 `restraints`

限制以无量纲伪观测残差加入目标函数，每项必须明确给出 `sigma`。

```json
[
  {"type":"distance","atoms":["C1","O1"],"target":1.23,"sigma":0.02},
  {"type":"angle","atoms":["C1","N1","C2"],"target":120,"sigma":2},
  {"type":"similar_distance","pairs":[["C1","O1"],["C2","O2"]],"sigma":0.02},
  {"type":"planarity","atoms":["C1","C2","C3","N1"],"sigma":0.02},
  {"type":"adp_similarity","atoms":["C1","C2"],"sigma":0.004},
  {"type":"rigid_bond","atoms":["C1","C2"],"sigma":0.004},
  {"type":"adp_isotropy","atoms":["C1"],"sigma":0.01}
]
```

长度为 Å，角度为度，ADP 为 Å²。`similar_distance` 使用每个后续距离与第一个距离之差；`planarity` 使用对 SVD 拟合平面的有符号距离。其参数含义不能直接替换 SHELXL 的全部 SADI/FLAT 规则。`rigid_bond` 是沿原子间方向的标量 Hirshfeld 限制，不是增强 RIGU。

普通标签使用最近周期像。需要指定跨晶胞、对称关系或较长配体中的具体原子像时使用：

```json
{
  "type": "distance",
  "atoms": ["C1", {"label":"O1","symop":"-x,y+1/2,-z","translation":[1,0,0]}],
  "target": 1.45,
  "sigma": 0.02
}
```

显式 `symop` 必须属于结构实际对称群；`translation` 是三个整数。指定了显式像后按该像的实际位移计算，不再自动选择最近像。

### 结果解释

`RefinementResult` 的主要字段为 `structure`、`scale`、`twin_fractions`、`metrics`、`history`、`parameters`、`parameter_names`、`covariance`、`standard_uncertainties`、`singular_values`、`nullspace`、`gauge`、`warnings` 和完整的输入配置。

`metrics` 包含工作集的 `r1_all`、`r1_gt`、`wr2`、`gof`、`n_work`、`n_free`、`n_parameters`、`jacobian_rank`、`diffraction_jacobian_rank`、`identifiable` 及 `free` 子字典。`history` 记录初值和各个外层权重循环，不是逐个内层优化步的轨迹。

原生雅可比使用中心三点有限差分，求解采用有界信赖域最小二乘。一般位点的各向异性 U 用 Cholesky 参数化，特殊位点用对称不变矩阵对数参数化。`u_min/u_max` 不等同于最终各向异性张量的逐特征值上界。

协方差以给定晶胞、散射模型、固定参数、规范和最终冻结权重为条件。`absolute` 使用绝对权重的局部曲率；`reduced_chi2` 再乘工作集的约化卡方。秩亏时 `covariance=None`，另行返回零空间与法矩阵伪逆；不能把伪逆当作缺失方向上的零方差。固定量、规范消去量和涉及活动边界的物理参数标准不确定度为 `None`。

### 留出数据

```python
from ariadne_xtal.refinement import grouped_holdout_flags

data.free_flags = grouped_holdout_flags(
    structure, data, fraction=0.10, seed=1729,
    twin_laws=None,
)
fit = refine(structure, data)
```

等价反射、Friedel 对及孪晶相关反射作为同一划分单位。精修前会检查这些关系是否被拆到工作集和留出集两边。另一接口 `validation.make_holdout()` 返回带标记的新 `Reflections`，使用稳定哈希划分，适合后续添加反射时保持分组规则。

留出反射不参与参数优化、初始比例估计、默认差值图和相位检索。反复查看留出结果并选择模型会消耗其独立验证作用；最终外部评估集应另行保留。

## 5. P1 候选求解

```python
from ariadne_xtal.phasing import PhasingOptions, solve_p1

result = solve_p1(
    cell, data, elements=["Co", "O", "O", "C", "C", "C"],
    options=PhasingOptions(trials=8, iterations=800, seed=1729, keep=3),
    scale=1.0,
)
```

| 选项 | 默认 |
|---|---:|
| `trials`, `iterations`, `seed` | 8、800、1729 |
| `threshold_sigma` | 1.1 |
| `oversampling` | 1.2 |
| `minimum_peak_distance` | 0.7 Å |
| `sharpen` | `True` |
| `keep` | 3 |
| `algorithm` | `"charge_flip"`；另可选 `"atomic_projection"` |

元素列表按整个晶胞计数，长度为 1–256。模块将重复观测和 Friedel 强度合并为实密度所需的振幅，使用多起点循环后提取峰，按峰强与给定元素的原子序数作初始分配。每个候选包含 `structure`、`trial`、`projection_residual`、`grid_shape` 和 `algorithm`。

返回的是供后续精修和化学检查的候选；投影残差不是解正确的概率。该方法不进行一般空间群发现、孪晶解卷积或绝对构型判定。分数坐标的平移或整体反演等价性应在比较解时处理。

## 6. 密度图与绘图

```python
from ariadne_xtal.maps import difference_map, model_map, patterson_map
from ariadne_xtal.visualization import create_dashboard, static_figures

rho = difference_map(fit.structure, data, fit.scale, kind="fo-fc")
rho.save_npz("difference.npz")
rho.save_ccp4("difference.ccp4")
peaks = rho.peaks(n=20, min_distance=0.8)

create_dashboard(
    fit.structure, data, "dashboard.html", scale=fit.scale,
    history=fit.history, covariance=fit.covariance,
    parameter_names=fit.parameter_names, density=rho,
)
static_figures(fit.structure, data, "figures", scale=fit.scale, density=rho)
```

`difference_map` 可选择 `fo-fc` 或 `2fo-fc`，默认排除留出反射。采用当前模型相位、实密度 Friedel 共轭平均、缺失系数置零；没有 sigma-A 权重或孪晶解卷积。针对已声明孪晶模型，agent 不生成未经解孪晶的差值电子密度图。直接 Python 调用时，应由调用者保证所选图的物理含义。

`DensityMap.values` 为三维数组，`cell` 保留实际晶胞。完整网格按 P1 写入 CCP4，因为请求的对称展开已在网格内完成。普通密度单位为 e/Å³，Patterson 为 e²/Å³。NPZ 保存类型、单位和元数据。

交互 HTML 内嵌 Plotly，不请求网络。三维位移椭球默认包络 50% 的高斯位移概率，其半轴为：

\[
a_i=\sqrt{\chi^2_3(0.5)\,\lambda_i(U)}.
\]

该椭球描述原子位移分布，不是原子平均坐标估计的置信椭球。图中的连线由几何邻近关系生成，不推断键级或化学价态。

## 7. 异常散射与绝对结构

```python
from ariadne_xtal.absolute import with_anomalous_factors, friedel_quotient

anomalous_model = with_anomalous_factors(structure, wavelength=0.71073)
diagnostic = friedel_quotient(anomalous_model, data, minimum_sum_significance=5.0)
```

Cromer–Liberman 适配器明确限制波长为 0.16–2.67 Å、原子序数为 3–92。需要 H 或超出范围的元素时，应直接提供适当散射项。

`friedel_quotient` 对固定结构拟合 (Q_{obs}=(1-2x)Q_{calc})，使用已测的正负 hkl 对及独立误差传播。返回 `x`、`standard_uncertainty_conditional`、配对数量、卡方和逐对数据；`x` 不截断到 0–1。其标准不确定度未包括结构协方差或共享吸收/比例误差，不是完整绝对结构统计判定。

需要联合优化结构和反演畴比例时，向 `refine()` 提供 (-I_3) 孪晶律并启用 `refine_twin_fractions`。

## 8. 结构反馈研究接口

```python
from ariadne_xtal.feedback import StructuralPrior, deblend_profiles

prior = StructuralPrior(
    mean=predicted_intensities,
    covariance=prediction_covariance,
    source_run="previous_crossfit_run",
    excluded_group_ids=["overlap_group_17"],
)
estimate = deblend_profiles(
    design=profile_matrix, observations=pixel_values, sigma=pixel_sigmas,
    group_id="overlap_group_17", prior=prior,
    strength=0.1, maximum_information_fraction=0.25,
)
```

该接口求解带非负强度约束的加权高斯轮廓问题，并可加入交叉拟合结构先验。程序检查目标组位于 `excluded_group_ids` 中，并限制先验相对于数据信息的强度。调用者负责真实构造排除该组的上游训练/精修流程；一个字符串声明本身不能证明数据独立性。

原始观测不被改写。数据轮廓矩阵秩亏时，程序拒绝用结构先验把不可分辨组报告成实验可辨识的强度。该接口没有实现原始探测器图像读取、指标化、积分或吸收校正管线。

## 9. Agent 请求协议

所有请求含 `action`，可带调用者的 `id`。每种动作只接受下表列出的字段，多余字段会报错。

| `action` | 必填 | 可选 |
|---|---|---|
| `capabilities` | 无 | 无 |
| `inspect` | `structure` | `hkl`, `spacegroup_candidates` |
| `refine` | `structure`, `hkl` | `options`, `restraints`, `constraints`, `graphics` |
| `solve` | `cell`, `elements`, `hkl` | `options`, `scale` |
| `maps` | `structure`, `hkl` | `scale` |
| `absolute` | `structure`, `hkl` | `options` |
| `legacy` | `engine`, `input` | `timeout`, `extra_args`, `auxiliary_paths` |

```python
from ariadne_xtal.agent import AgentWorkspace

workspace = AgentWorkspace("/absolute/path/to/project")
response = workspace.dispatch({
    "action": "refine",
    "structure": "examples/demo_start.json",
    "hkl": "examples/demo_reflections.json",
    "graphics": False,
})
```

默认每个输入文件最多 200 MiB，结构最多 500 个独立原子、反射最多 100,000 条；agent 将精修 `max_nfev` 限为 5,000，相位循环总数 `trials × iterations` 限为 20,000。底层模块另有网格及候选规模限制。差分构造雅可比仍会产生额外函数评估，不能把 `max_nfev` 当作总结构因子计算次数。

输入须位于工作区内；输出为新建的 `runs/` 子目录。原生操作使用输入快照，并检查输入文件在运行中未变化。清单的 `scientific_acceptance` 保持 `not_assessed`。可选格式/绘图导出问题记录在 `artifact_warnings`。

单个任务返回：

```json
{
  "ok": true,
  "run_id": "refine_...",
  "manifest": "runs/refine_.../manifest.json",
  "result": "runs/refine_.../result.json",
  "summary": {"success": true, "metrics": {}}
}
```

工作目录边界和无 shell 调用用于隔离输入输出，不是操作系统安全沙箱。被允许的外部可执行文件仍以本机进程权限运行。

## 10. 外部后端 API

```python
from ariadne_xtal.compat import run_legacy

manifest = run_legacy(
    "shelxl", "/absolute/path/to/shelxl",
    "crystal/sample.ins", "runs/official",
    timeout=300, extra_args=[], auxiliary_paths=None,
)
```

API 不搜索 PATH，调用方提供绝对可执行文件路径。RES 可按原字节复制为同名 INS。原始文件所在目录树内的 include、HKL/FAB 和明确附属文件按原布局复制，绝对或 `..` include 路径被拒绝。附属文件不能超出该输入项目目录。

`status` 为 `completed`、`process_failed`、`timed_out` 或 `start_failed`，另有 `returncode`、`elapsed_seconds`、`working_directory`、`manifest_path` 及输入输出哈希。`scientific_validation_performed` 固定为 `False`，表示包装层不把进程退出状态解释为结构正确性。

通过 agent 使用该 API 时，须在创建 `AgentWorkspace` 或启动服务时传入 `legacy_executables={"shelxl":"/absolute/path/to/shelxl"}`。请求不能自行指定新的可执行文件。
