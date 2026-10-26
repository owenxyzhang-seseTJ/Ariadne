# 重叠积分：原理、实现与接入 · Ariadne 0.5.0

本版实现**读取原始像素 → 同曝光重叠分组 → 联合轮廓积分 → 完整协方差 → 校正后观测组 → HKLF 5**。2026年10月26日已运行新增方法的针对性数值验证；真实衍射数据完整流程仍未验证。位置扰动、解析Gaussian体素积分、完整Poisson–Gaussian及边界区间见 [algorithm-upgrades.md](algorithm-upgrades.md)。

## 原理与选择

使用原始像素的线性观测模型：

\[
\mathbf y=\mathbf P\mathbf I+\mathbf B\boldsymbol\beta+\boldsymbol\epsilon.
\]

P 每列为一个完整归一化反射轮廓，I 为该贡献的总计数；B 为局部常数或三维背景平面。完整预测列表决定贡献者，不使用强点表代替预测，否则弱重叠贡献会漏掉。

选用校准噪声下的加权最小二乘：保留弱反射的负强度，支持积分型探测器的读出噪声，不假定所有图像都是纯 Poisson 光子计数。已知增益/读出方差时可 IRLS 更新预期方差。它适合固定轮廓的线性求解，不是所有噪声/轮廓条件下的普适最优方法。

先对加权 B 做 QR，消去背景干扰，再对剩余轮廓矩阵做 SVD。这样不构造像素数平方大小的投影矩阵，也不直接求病态正规矩阵的逆。强度协方差包含背景参数的不确定性；profile/geometry 的额外误差仍需另建模型，本版不会把它伪称已包含。

部分重叠且可辨识时，输出各贡献强度及**非对角协方差**。总强度的方差为：

\[
\operatorname{Var}(I_\mathrm{sum})=\mathbf1^\mathsf T C_I\mathbf1.
\]

完全重合时，轮廓矩阵可能秩亏；检验某个强度对比是否与零空间正交。各分量不可辨识但和可辨识时，返回 `total_only`，不发布伪逆得到的任意拆分值。总和也不可辨识时保留 `unresolved` 和原始 patch。

这与从多矩阵同时积分保留 single/composite observations 的文献路线一致；不同角度的相近 q 不能作为同曝光组。原生 DIALS 确有 detector-space deconvolution，本机 3.30.1 源码也有多轮廓拟合；此前“DIALS 未支持重叠”的笼统表述已纠正。原生功能存在不等于提供了这里所需的完整跨分量协方差与 HKLF 5 交接。

## 两条实际路线

优先用已有成熟原生模型，参数文件为 `assets/native-overlap.phil`：

```bash
scxrd-env dials.integrate multi_domain.expt reference.refl /absolute/skill/assets/native-overlap.phil
```

记录 native 日志、轮廓、未积分点和重叠处理。该命令不是本次已执行的测试。原生结果不能直接按 nearest-q 复制成 HKLF 5。

需要明确总观测/完整协方差或原生路线受限时，使用本版独立 QR/SVD 积分器：

```bash
scxrd-env python scripts/learn_profile_library.py isolated_reference.refl calibration.json library
scxrd-env python scripts/overlap_prepare.py multi_domain.expt predicted.refl library/profiles.json patches
scxrd-env python scripts/overlap_integrator.py patches/patches.json joint --irls
scxrd-env python scripts/correct_overlap_groups.py joint/integrated_groups.jsonl corrections.json corrected
scxrd-env python scripts/export_hklf5.py corrected/corrected_observations.jsonl twin.hkl
```

脚本路径需替换为本 skill 内绝对路径。顺序是实际数据处理关系；安装插件时不执行。library、calibration 和 corrections 是真实实验/软件证据，不能从示例数值推测。

## 完整多晶畴预测

`dials_tools.py build-domain-experiments` 将确认过的 A 模型复制到各个原始 sweep，以独立晶体模型和共享仪器几何构建 `multi_domain.expt`。再用 DIALS 预测全部反射。

模型文件额外要求：`phase_relation=same_phase_supported`、`common_structure_basis`、`phase_basis_provenance`。每域 A/HKL 必须已经换到共同结构基准，不能只修改文字字段。不同物相不能共用一个 SHELXL 孪晶模型。保留 `domain_map.json` 的 experiment-to-domain 对应，不能把 sweep 当晶畴。

## 轮廓学习契约

`calibration.json` schema 为 `scxrd-profile-calibration/1`，必需字段：

| 字段 | 含义 |
|---|---|
| isolated_source_rows | 真实参考表内已确认独立、不重叠的反射行 |
| isolation_provenance | 隔离判据/完整预测检查来源 |
| background_provenance | shoebox.background 的真实拟合来源 |
| calibration_support_complete | 参考轮廓支持区域经原图检查完整，必须为 true |
| noise_model | gain、readout_variance、source；可给计算用 variance_floor |
| basis | 共同结构基准及索引约定 |
| experiment_to_domain | 字符串实验编号到晶畴编号的映射 |
| complete_prediction_provenance | 多域完整预测来源 |

模板学习用高 SNR 的孤立点，并记录未满足条件的**参考点**；不因此删除最终弱反射。训练把背景扣除后的正信号归一化、对齐后平均，保留源行和建模近似。单帧参考不能证明完整 rocking profile，默认要求至少两帧；若真实仪器 profile 更窄，应采用经过支持检查的其他模板来源。

library schema `scxrd-profile-library/1`，每个 entry 指明 experiment_id、panel、template NPY、covariance_px_xyz、noise_model、provenance。也可提供有据的 Gaussian covariance 替代 empirical template。0.5.0 可用 `--local-grid 3 3 4` 学习向 pooled 收缩的局部模板，像素适配器按预测位置做非负 trilinear 混合，缺参考区域使用 pooled。详见 [source-extensions.md](source-extensions.md)；局部模型仍是条件经验峰形，没有额外传播模板误差。

## 像素与观测组

同 imageset、扫描范围和 panel 的预测轮廓支持框组成图；扫描轴排序后仅比较相交的活跃框，形成连通组。背景 halo 也参与分组，避免另一组的峰被当作背景。每组一次读取联合 ROI，原始帧使用有界 LRU cache。

NPZ 包含 `data/variance/valid` 三维 ZYX 数组和 `profiles` 四维 KZYX 数组，不允许 pickle。轮廓保留完整归一化，裁剪/掩膜后不重新归一化；有效质量不足时建议扩窗、检查遮挡/过载，而不是补成“完整峰”。

patch/component 预算超限的预测记录保存在 manifest，不能无声丢掉。模板、几何、所有贡献者和不确定度来源一起存档。固定 profile 模型失败时优先使用局部模板/分段几何，再扩大联合 ROI。

## 校正和 HKLF 5

积分输出是 detector counts，**还不是校正后强度**。corrections schema 为 `scxrd-group-corrections/1`，包含 basis、integration_provenance、correction_provenance、domain_count、complete_contributor_audit、contains_lp_scaling_absorption、scale_convention 和 groups。

`scale_convention` 必须为 `common_across_domains_keep_twin_fraction`。不能分别把各晶畴强度归一化到同均值，消掉真实体积分数。

每组 correction 的 `factors` 与贡献者顺序一致，可带完整 `factor_covariance`；未提供时明确 `factor_uncertainty=treated_as_fixed`。若校正来自同一批强度，提供强度—校正交叉协方差。采用一阶误差传播；跨观测的尺度相关性另存原校正模型，HKLF 5 无法编码全部相关性。

全部分量可辨识时，可以分别校正后求总和，并传播完整 C_I。只有原始总强度可辨识时，校正器仅允许相同的共同校正因子；不同因子不能通过强拆分或平均补救，应交由 `global_scale.py` 的跨等效联合估计，或使用适用后端。那里这些组参与拟合但标记为 fit-only，不出导分量强度。

有 profile/background misfit、未收敛、未知校正或无法表达的总量时，保存原观测并写拒绝原因，不生成虚假的 corrected records。最终 HKLF 5 每组只对应一次真实总观测，不重复导出同一组的各个分量作为独立测量。

## 高计算量的补救

1. 分域：`search_domains.py --heavy --max-index-attempts 128 --max-seconds 14400`，多轮分层抽样、smart/naive 路线、FFT3D/FFT1D 轮换，失败候选不剥点。
2. 积分：扩大 profile 支持范围/背景 halo/patch budget；有据替换局部 empirical template，增加 IRLS 迭代。每次保留独立输出，选择依据是残差、覆盖、协方差和等效点一致性。
3. 完全重合的不可辨识分量不靠加算力伪造，保留可测总量进入 composite refinement；若校正模型仍不足，尝试等效反射联合模型或用户已有的其他后端。

当前 brute-force 分域调度器已实现；积分扩窗配置与 kernel 已实现，但未做自动 profile/geometry 模型选择。全局缩放吸收见 [global scaling](global-scaling.md)，为共享经验低阶模型，不是完整 TWINABS 等价实现。

## 主要依据

- [Sevvana et al. (2019): non-merohedral twinning](https://journals.iucr.org/d/issues/2019/12/00/rr5182/index.html)：多矩阵同时积分、composite observations 与校正的不同处理需求。
- [Leslie (2006): integration and empirical profiles](https://journals.iucr.org/d/issues/2006/01/00/ba5082/)：标准轮廓学习、加权拟合与轮廓失配误差。
- [DIALS integrate documentation](https://dials.github.io/documentation/programs/dials_integrate.html)：detector-space deconvolution 参数。
- 实际本机 3.30.1 的 `algorithms/integration/fit/fitting.h` 与 `profile_model/gaussian_rs/model.py`：多轮廓拟合与可配置原生接口。源码阅读不代表本次运行验证。
