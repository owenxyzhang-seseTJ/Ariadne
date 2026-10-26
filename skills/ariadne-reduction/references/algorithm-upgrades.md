# 位置扰动、计数似然与多晶格 EM

这些方法通过显式选项启用。现有默认方法保留。测试使用已知真值算例与真实 DIALS 数值接口；真实孪晶原始图像的性能、跨仪器适用性及完整 SyntheticTwinBench 尚未验证。

## 位置扰动与解析体素积分

先运行 `overlap_prepare.py MODEL.expt PREDICTIONS.refl LIBRARY.json NEW_PATCHES --position-derivatives`，再运行 `overlap_integrator.py NEW_PATCHES/patches.json NEW_FIT --fit-positions`。MCP 对应 `build_overlaps(position_derivatives=true)` 和 `joint_fit(fit_positions=true)`。

对每个固定轮廓 P，保存对中心 XYZ 的有限差分导数 D，步长 0.001 px。拟合模型为 `y = B beta + sum(I_k P_k + D_k m_k)`，其中 `m_k = I_k delta_k`。这是局部一阶模型。QR 消去全部背景与位置 nuisance 方向，SVD 判断强度 contrasts；输出的强度协方差已边缘化这些 nuisance。只有 I 和对应 moment 均可辨识且 I 不接近零时，才输出 `delta = m/I`，位移方差使用包含相关项的 delta method。

默认 `--max-position-shift .3` px；超过此值设置质量标记，下游校正不会继续把它当可靠强度。此模型不重新采样非线性轮廓，不传播独立仪器标定及轮廓形状误差。仅适用于 `signed-wls`，可与其 IRLS 组合。完全重合分量不会因加入导数而变为可辨识。

对角 Gaussian 协方差使用一维 CDF 差值的乘积精确积分每个体素，正尾使用生存函数避免相消。相关 Gaussian 使用与条件宽度相关的中点求积；超过每轴 16 点的预算明确拒绝，不静默截断到欠采样值。经验轮廓仍使用现有插值。

## 非负边界区间与 Poisson–Gaussian

`overlap_integrator.py PATCHES.json NEW_OUTPUT --estimator poisson-gaussian --confidence-level .95` 启用 `Y = gain * N + Normal(0, readout_variance)`，其中 N 服从 Poisson。`noise_model` 需要真实 `source`、正有限 `gain`、非负有限 `readout_variance` 和 `offset_removed=true`。正读出方差允许负 ADU；零读出退化到精确整数计数 Poisson。原 `poisson` 方法仍额外要求 `poisson_counts_verified=true`。

实现用 logsumexp 与相邻项比率控制卷积两侧遗漏尾部，相对尾部容差 1e-12；没有高计数 Gaussian 替代。每个平移像素求和预算 100000 项，每次 likelihood evaluation 2000000 项，预算包含重复扩展成本；超限明确失败。密度、score、NLL 截断误差界及实际项数保存在 `likelihood_diagnostics`，误差界不包含浮点舍入。PG 的 `chi2` 兼容键保存 -2 log density，不能按 chi-square/deviance 解读，`reduced_chi2=null`。

所有非负方法为可辨识分量和总量生成 `profile_likelihood_interval`：固定目标强度时重新优化其他非负分量。区间下限可为零；不可辨识、MLE 未收敛或 profile 优化失败时不输出伪端点。默认 95% 区间使用 chi-square(1) 渐近阈值，低计数和边界处不保证精确覆盖率。背景先由背景区域 WLS 估计再冻结，区间不包含背景估计、轮廓形状和仪器标定的不确定性。

master JSONL 保存非对称区间。现有校正和 HKLF5 使用 Gaussian sigma，无法编码边界区间，因此边界的质量保护继续保留。PG 协方差是 moment-weighted Gaussian 代理，不宣称是精确 Fisher 信息。

## 单源质心多晶格 EM 与留出比较

`joint_geometry_refine.py MODEL.expt SPOTS.refl COMPATIBILITY.json NEW_OUTPUT --method em --centroid-selection SELECTION.json`。可追加 `--compare-models ALTERNATIVE.expt ...`，预先声明候选晶格集合并分别在同一训练行上拟合。默认锚点分支仍使用 `--method anchors`。

`SELECTION.json` 包含 `single_source_source_rows`、`composite_source_rows` 和真实 `provenance`；剩余行保留为未知并排除。几何上有多种可能归属的单源质心可以参加 EM，实际复合峰不能当成互斥源质心。可提供 `candidate_generation_source_rows`、`calibration_source_rows`，两者须均属于扫描块训练分区；缺少上游训练来源时，模型数比较保持 unresolved。

E 步使用每个原始扫描的完整初始可观测预测集合，在 mm/mm/未折叠 rad 坐标上求完整协方差 Gaussian 与均匀 outlier 的归一化密度。每个预测的先验质量相同；增加无支持预测会分摊概率质量。它不按局部近邻数重新归一化，不把 compatibility 当 posterior。初始预测的 HKL/分支身份冻结，位置随每次 objective evaluation 更新；这属于局部支持近似。

M 步用 DIALS 保持对称性的晶体取向/晶胞参数和原生预测，仪器参数固定。完整协方差在外层 Cholesky 白化，以精确充分统计汇总每个预测的加权目标，避免建立 observation × prediction × parameter 的巨大 Jacobian。训练观测似然下降时拒绝更新。有效支持不足、参数秩不足、碰到局部边界、优化/预测预算耗尽均保留 unresolved。

扫描块分区先于拟合确定，所有候选训练结束后才计算留出密度。输出逐行 log density、对第一候选的增益、按块 jackknife 标准误与 Student-t 95% 区间。少于三个有分数的留出块不作确定判断。比较保留所有模型，不自动改写 K；它是有条件的预测比较，不是正式 Bayesian evidence，也不代替上游盲测、等价/超胞和孪晶律判断。

默认预算：8 个 EM 迭代、每个 M 步 20 次函数评价、500 次原生预测调用、300 秒、20000 个完整预测、2000000 个 observation/prediction 对、最多 4 个预先声明的假设。原生数值差分的预测调用也计入预算。预算有限时减少数据量或收紧有据候选范围，不静默截断完整预测。

MCP `start_analysis(analysis='joint_refine', refinement_method='em')` 的 inputs 顺序为 MODEL.expt、SPOTS.refl、COMPATIBILITY.json、SELECTION.json，后接最多三个候选 EXPT。位置和积分选项与不同 analysis 错配时明确拒绝。

## 验证入口

在匹配 DIALS 3.30.1 的环境中运行 `python -m unittest discover -s tests -v`。不需要下载数据或启动 MCP 服务。测试包含解析概率、中心扰动、协方差、不可辨识分量及相应的似然/EM 检查。测试不会执行文件散列校验。

## 本次验证记录（2026年10月26日）

- 正常运行配置下，`python -m unittest discover -s tests -v`：35 tests OK，7.866 秒。包含 4 个位置/概率检查、21 个受约束似然检查、10 个 EM 检查；DIALS 环境下的原生适配检查实际执行。
- 双峰真实位移算例：固定位置的分量强度误差 -2.814% / +9.353%，一阶位置 nuisance 后 -1.395% / +1.859%。单个已知真值算例，不代表真实数据普遍增益。
- 双晶格 DIALS：6240 个完整预测、18 个参数、rank=18，位置 RMSD 从 [0.0070651, 0.00189798, 0.000391690] 降至 [1.30691e-8, 4.16547e-9, 6.23230e-11]（mm/mm/rad）；仪器字典前后相等。无噪声合成数据可检验实现，不代表实验精度。
- 最终留出改变不影响拟合矩阵；额外无支持晶格的留出平均 log density 增益为 -0.405678，支持不足仍输出 unresolved。
- MCP 固定命令路由检查：EM/备选模型、PG/置信度、位置导数和拟合参数通过，无效组合拒绝；未启动真实 MCP 客户端会话。
- 受约束数值测试在 `-W error` 下通过；完整测试在相同 warning 策略下首次以代码 139 退出。随后正常配置打印 DIALS/SWIG 的 DeprecationWarning 并全量通过。代码 139 的底层原因尚未确认，不把严格 warning 配置称为通过。

尚未重装已部署的 ariadne Python 包、未重启 MCP、未运行实测原始帧的完整对比。源码修改并不表示正在运行的服务已加载新接口。
