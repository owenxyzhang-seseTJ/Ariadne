# 数学与 API 约定

本文规定 Ariadne-Xtal 0.1.0 各模块共享的坐标、单位、数据和结果语义。完整调用示例见 [API 文档](docs/API.md)，原生与官方后端的功能关系见 [兼容性说明](docs/COMPATIBILITY.md)。Python 包名为 `ariadne_xtal`，基础环境要求 Python ≥3.10。

## 1. 坐标与晶胞

`Cell(a, b, c, alpha=90, beta=90, gamma=90)` 使用 Å 和度。`cell.matrix` 以直接基矢为列，记为 \(A\)；`cell.reciprocal` 为 \(A^{-\mathsf T}\)，不含 \(2\pi\)。因此：

\[
\mathbf r=A\mathbf x,\qquad
\mathbf q=A^{-\mathsf T}\mathbf h,\qquad
d_{\mathbf h}=\lVert\mathbf q\rVert^{-1},\qquad
s^2=(\sin\theta/\lambda)^2=\lVert\mathbf q\rVert^2/4.
\]

`to_cartesian()`、`to_fractional()`、`d_spacing()` 和 `stol_sq()` 接受末轴长度为 3 的数组。`minimum_image(delta_frac)` 返回具有最短笛卡尔长度的周期分数位移，通过有界搜索求解一般非正交晶胞的最近像。

## 2. 原子与对称

`Atom` 包含 `label`、`element`、`frac`、`occupancy`、`u_iso`、`u_aniso`、`f_prime`、`f_double_prime` 和三个精修标志。分数坐标为长度 3 的数组；`u_aniso` 为 **3×3 笛卡尔 U 张量，单位 Å²**。完整占据的物理位点占有率为 1，包括特殊位置。

`Structure(cell, atoms, spacegroup="P 1", wavelength=0.71073, sym_ops=None)` 保存独立位点。显式 `sym_ops` 优先于 H–M 名称，必须构成完整群并与晶胞度量相容。标签不区分大小写地要求唯一。`structure.operations` 返回 Gemmi 对称操作列表。

`unique_sites(structure, atom)` 返回唯一对称像的 `(frac, U_cart)` 列表。张量随每个操作旋转；特殊位置上到达同一位点的张量按稳定子投影。对特殊位置的结构因子求和采用唯一像和物理占有率，不重复乘一般位置重数。

标准 CIF/SHELX Uij 与内部张量满足：

\[
U_{\rm cart}=A D U_{\rm CIF} D A^{\mathsf T},\qquad
D=\operatorname{diag}(|a^*|,|b^*|,|c^*|).
\]

## 3. 结构因子与孪晶

`structure_factors(structure, hkl)` 返回复数数组。采用中性原子 IT92 散射因子及明确给定的异常散射项，傅里叶相位为 \(\exp(2\pi i\mathbf h\cdot\mathbf x)\)。Debye–Waller 振幅因子为：

\[
\exp(-2\pi^2\mathbf q^{\mathsf T}U_{\rm cart}\mathbf q),
\]

各向同性时为 \(\exp(-8\pi^2Us^2)\)。反变换电子密度采用负相位和 \(1/V\) 归一化。

`calculated_intensities(structure, hkl, scale=1.0, twin_laws=None, twin_fractions=None)` 计算：

\[
I_{\mathbf h}^{\rm calc}=k\sum_d\alpha_d|F(T_d\mathbf h)|^2.
\]

`scale` 是强度比例。`twin_laws` 不包含单位变换；`twin_fractions` 包含单位畴且置于首位，非负并总和为 1。矩阵作用于倒易列向量，数组用 `hkl @ T.T`。原生孪晶律限于保持倒易度量的整数幺模矩阵，表示完全重叠的畴强度和。

`systematic_absences()` 返回消光标记；`rank_spacegroups()` 返回候选群的矛盾诊断。消光一致不构成空间群唯一判定。

## 4. 观测与精修结果

`Reflections(hkl, intensity, sigma, batch=None, free_flags=None)` 要求非空 N×3 整数指标、有限测量强度以及严格为正的标准不确定度。负强度保留。`free_flags=True` 表示留出观测。

`refine(structure, reflections, options=None, restraints=None, constraints=None)` 返回 `RefinementResult`，不原地修改输入。目标使用全部工作集 F²；留出集不参与参数优化。结果包含模型、强度比例、孪晶比例、计算强度、参数、指标、历史、约束/限制、规范信息和不确定度诊断。

精确约束类型为 `equal_xyz`、`equal_adp`、`equal_occupancy`、`occupancy_sum`。限制类型为 `distance`、`angle`、`similar_distance`、`planarity`、`adp_similarity`、`rigid_bond`、`adp_isotropy`；每项明确给出 `sigma`。

`success` 表示数值收敛，`metrics.identifiable` 表示当前雅可比满列秩。协方差以给定晶胞、模型、规范和最终冻结权重为条件。秩亏时协方差为空，另行报告零空间与法矩阵伪逆；固定量、规范消去量及边界参数不能解释为具有零实验不确定度。

## 5. 文件与执行接口

`read_structure()` / `write_structure()` 支持 JSON、CIF 和明确的 INS/RES 结构子集。`read_reflections()` / `write_reflections()` 使用 JSON 或 HKLF4；JSON 保留原生字段及浮点精度，HKLF4 保留负强度和原始批次。

`ShelxDocument` 将原始文本保留与数值转换分开。`write()` 写出原文；`to_structure()` 拒绝未翻译的科学指令。新的结构 RES 导出不代表保留原始精修配方。SHELX SOF 转换为物理占有率时乘以一般位置重数与实际位点重数之比。

`run_legacy(engine, executable, input_path, output_dir, timeout=300, extra_args=None, *, auxiliary_paths=None)` 在新目录中，以无 shell 的参数数组运行明确指定的独立安装引擎，保留输入字节并记录文件哈希。没有从原生计算到官方后端的自动切换。目录隔离不是操作系统安全沙箱。

## 6. Agent 与绘图语义

Python、CLI、JSONL 和 MCP 共用同一数值 API。Agent 请求仅调用已声明动作，输入限制在工作区内，输出记录任务、环境和文件来源。任务执行完成、数值收敛和结构的科学接受分别记录，不相互替代。

密度图明确保存系数、比例与单位约定；默认差值图排除留出观测。位移椭球由 U 张量及三自由度卡方分位数确定，描述原子位移分布，不是平均坐标估计的置信椭球。
