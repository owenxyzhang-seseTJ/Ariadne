# Ariadne 0.5.0：附件方法论的接入范围

参考用户提供的 `Agent_Native_Probabilistic_Multilattice_Methodology.md`。附件是研究方案；其中的命令式表述、自动审查、测试和未来研发建议不构成新的执行授权。附件的 checksum_manifest 改为路径/大小/时间与输入输出记录。本地数据和禁止 SHA-256 校验的约束保持；2026年10月26日已按后续优化任务运行针对性数值验证，范围与新增选项见 [algorithm-upgrades.md](algorithm-upgrades.md)。

## 实现范围与验证边界

| 附件主张 | 本版实现 | 边界 |
|---|---|---|
| 候选池、多假设与回退 | immutable hypothesis JSON、parents、证据引用和 append-only event log | phase_relation 为声明证据，不是自动孪晶鉴定；继续复用原 domain_manager 等价检查 |
| 非互斥软解释 | q residual + DIALS 全预测的 mm/mm/rad Mahalanobis compatibility | exp(-D²/2) 非标定后验；无强度/体积分数含义 |
| 联合几何更新 | unique anchors 默认分支；显式单源质心 EM，DIALS U/B 参数和动态预测，共同仪器固定 | EM 仅针对独立单源质心；复合/未知质心排除，不是完整多源生成模型 |
| 留出评价 | 原 DIALS candidate scorer 改扫描块留出；joint refiner 保存固定留出行 | 候选生成可已看过留出点，非端到端盲测 |
| 全反射预测、一次像素观测 | native predict + 原 overlap_prepare 的共同 ROI 连通组 | bbox/halo 是保守分组，不等于 profile 不可辨识 |
| NNLS/Poisson/Poisson–Gaussian | fixed-background 非负拟合、完整有界 PG 卷积、profile-likelihood 区间 | 背景估计误差未传播；区间用渐近阈值，边界不能变成 Gaussian sigma |
| 可辨识性/复合强度 | SVD/nullspace contrasts、weighted profile correlations、A/B/C 标记 | 数值阈值为配置假设，未基准校准；不发布任意拆分 |
| covariance/partiality/导出 | corrected component covariance、有效 profile mass、禁止双重校正、HKLF5 行映射和损失说明 | HKLF5 无法编码全部 covariance；保存 master JSONL |

## 软几何匹配

`soft_multi_indexer.py` 输入顺序：观察 EXPT、spots REFL、多晶格模型 EXPT、全预测 REFL、标定 JSON、新输出目录。

标定 JSON 必须提供以下真实字段，不能复制示例数值充当标定：

- `model_covariance_mm_rad`：3×3 PSD 预测误差；`model_covariance_source`。
- `centroid_covariance_source`：观察质心误差来源；适配器读取 measured mm variance 或由真实 px variance 转换，不接受 DIALS 补出的单位方差。当前观察协方差采用对角近似。
- `q_tolerance_invA` 和 `q_tolerance_source`：倒易匹配筛选的来源；不是自动校准概率。
- `instrument_geometry_provenance`：共同仪器几何的标定来源。
- `prediction_scope=all_observable_reflections`、`complete_prediction_provenance`：由所有模型预测，而非只有 strong spots。
- `experiment_to_lattice`：字符串实验 id 到稳定晶格 id 的映射；sweep/experiment 不自动等于晶畴。

MVP 为静态、单波长旋转模型，动态 crystal/beam/goniometer 需明确分段后处理。先按共同原始扫描与 panel 分区，检查仪器签名一致，以保守 KD-tree screen 查 native prediction，再检查 q 与 mm/mm/rad residual。原生预测包含 Ewald 几何；scan angle 不跨不同曝光做 modulo 2π 合并。缓存路径/几何/A，避免逐点重复构造图像路径列表。

输出逐 spot 的全部有限解释、compatibility、位置 log density、源行及 residual。搜索预算超限保留该 spot 和原因，不能只截取“最像”的若干模型。位置 log density 不被称为完整多源 spot likelihood。`compatibility`、像素计数贡献、晶畴体积分数三个量始终分开。

## 唯一锚点联合精修与比较

`joint_geometry_refine.py MODEL.expt SPOTS.refl COMPATIBILITY.json NEW_OUTPUT` 调用 DIALS RefinerFactory。仅把 unique-explanation anchors 编成一行一次观测，赋入对应实验/HKL；方差根据兼容度软加权。仪器几何固定，模型预测误差在 native objective 中采用对角近似。不确定的多个贡献仍在原表和 compatibility 中保存，进入后续共同像素拟合。

按 raw observed experiment + frame block 划分训练与留出，默认每块 20 帧。每实验锚点不足则失败，不把 ambiguous spot 复制到几个晶格填够数量。输出 refined.expt、training.refl 和 joint_refinement.json。后者列出固定 validation rows、训练权重来源和 parent model。

对 refined.expt 重新全预测，再用所有原始 spots 做软匹配。`compare_hypotheses.py PARTITION.json NEW_COMPARISON.json BASELINE_COMPAT.json ALTERNATIVE_COMPAT.json ...` 在完全相同留出行上比较 coverage、最佳 compatibility 与独立支持，输出多种解释比例、预算超限行与 gain。噪声/标定或观察表不同则拒绝比较。该指标是诊断，不是 Bayesian evidence；没有把非互斥 centroid scores 塞进伪 BIC。增加 K 的最终选择还需稳定性、等价/超胞检查、像素拟合增益和相关系证据。

## 四种像素估计器

`signed-wls` 保留原 QR/SVD 分支，有符号强度估计和包含 nuisance background 的条件协方差。

`nnls`：从 profile 显著区域之外的像素拟合低阶背景并冻结，使用 SciPy bounded least squares；I≥0，弱峰可在边界。固定背景的近似协方差不含背景估计、profile 或几何误差。背景像素不足/背景期望负值则明确失败。背景像素的 1% profile 支持界限为配置实现近似，不是已验证最佳划分。

`poisson`：在相同冻结背景/profile 上优化 Poisson negative log likelihood，要求 `noise_model.source`、正 gain、readout_variance=0、`offset_removed=true`、`poisson_counts_verified=true`，并检查有效 ADU/gain 为非负整数。拟合在 photon counts 中，输出 amplitude/covariance 换回 calibrated ADU 以维持校正尺度。不将 CCD 的负 ADU/读出噪声当成理想计数。此分支不是完整 Poisson–Gaussian detector likelihood。

`poisson-gaussian` 使用完整 Poisson 与读出 Gaussian 卷积，允许 offset-corrected 负 ADU；尾界和预算、profile-likelihood 区间详见 [algorithm-upgrades.md](algorithm-upgrades.md)。非负约束不会创造可辨识性；rank-deficient 分量不发布任意求解值。区间重新优化其他强度，背景冻结，阈值为 chi-square(1) 渐近值。非对称边界区间保存在 master JSONL，质量标记仍阻止将其当 Gaussian sigma 导出。

输出 full-profile amplitude、`profile_valid_masses` 和 amplitude×mass 的 observed estimate。当前是经验 finite-support 模型下的质量，并非额外独立测出的几何 partiality；不在 scaling 中重复除以它。新 correction group 需 `partiality_policy=already_full_profile_amplitude`。master JSONL 保存完整条件 component covariance，校正后也保存 transformed covariance；total-only 不伪造 component estimates。

HKLF5 twin export 要求 `phase_relation=same_phase_supported` 和 `phase_evidence_provenance`。不同晶相不可打包成一个孪晶结构。导出附 observation-to-line mapping 及 component/cross-observation covariance loss；源 JSONL、积分与校正模型必须一并保存。旧 JSONL 缺上述新证据时补充真实来源后才能导出，不能猜字段。

## Agent 工具与执行关系

MCP 现在有 9 个工具：原 capabilities/create_job/start_operation/inspect_job/read_artifact/record_acceptance/make_handoff，加 `start_analysis` 与 `propose_hypothesis`。native operations 新增 `predict`。`start_analysis` 的枚举包括 soft_assign、joint_refine、learn_profiles、build_overlaps、joint_fit、compare_hypotheses、export_hklf5，各绑定固定脚本及参数/输出契约，无 shell 任意执行入口。

实际循环：提议 hypothesis → native predict → soft_assign → joint_refine → new predict/soft_assign → fixed holdout compare → agent 根据证据接受或保存备选 → profile/overlap/joint_fit → 真实校正因子传播 → phase-aware HKLF5/可信 HKLF4 → solve-single-crystal。

假设及计算输出不覆盖 parent；events.jsonl 保存 host、时间、软件、operation 与 evidence 引用。worker PID 消失但无结果时读状态标记需审查，先检查日志/后代计算；不自动重放。科学阶段接受依赖证据，外部自动计算退出 0 不会自行获得接受。需要较多 domain-level 循环时显式增大 create_job 的每阶段 attempt budget，例如 12，仍有总时限和停止条件。

## 仍是研发目标

完整多源 spot 生成模型、正式 Bayesian K evidence、轮廓形状及独立仪器标定误差传播、盲测与跨仪器性能仍未实现/验证。全局 composite-aware scaling/absorption 已实现共享扫描尺度与低阶方向项，见 [global scaling](global-scaling.md)；它不估计独立 domain multiplier。现已提供单源 EM 与留出预测密度比较、非负边界渐近区间、局部一阶位置 nuisance 和强度协方差传播。SyntheticTwinBench 是附件建议，本次没有生成或运行模拟测试集。未因文档要求增加人工审批关卡；科学证据不足则保留未知，由已授权的 agent 流程处理。

## 可追溯原理来源

- [DIALS multi-lattice tutorial](https://dials.github.io/documentation/tutorials/multi_lattice_tutorial.html) 与固定版本源码 `lattice_search`、RefinerFactory、ReflectionPredictor。
- [Sevvana et al. (2019), Non-merohedral twinning](https://journals.iucr.org/d/issues/2019/12/00/rr5182/index.html)：独立/复合反射与结构层约束。
- [SciPy lsq_linear](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html)、[minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)：受约束优化实现接口。

上述文献支持公开原理/接口；新增方法已有针对性数值验证，尚无完整真实数据基准，不宣称算法创新或普遍性能优势已获证实。
