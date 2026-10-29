---
name: ariadne-reduction
description: Reduce Bruker or Rigaku single-crystal X-ray diffraction frames through spot curation, geometry checks, reindexing, multi-domain/twin diagnosis, integration and absorption correction to INS/HKL, then hand off to solve-single-crystal. Use for raw-frame reintegration or unmerged-observation rescaling; corrected HKL-only structure solution belongs to solve-single-crystal.
---

# Ariadne Reduction

用简体中文沟通。以原始衍射证据决定还原路线。默认在交接就绪后继续 `solve-single-crystal`，除非用户要求停在还原阶段。本技能是 AI 决策流程，脚本承担确定性辅助；需要真实、适用的积分和吸收校正软件。

优先读取 `ariadne` MCP 的 `capabilities`，据 `analyses` 的输入角色与适用选项决定调用，并用 `create_job → start_operation → inspect_job/read_artifact → record_acceptance` 管理真实任务。长任务返回 operation id 后再读取状态，避免同步等待整轮积分。MCP 不运行安装测试，不自动接受计算结果。复合重叠和分域使用下述脚本；MCP 的 native scale/export 仅用于适用的单相分支。

运行前读取 [source-extensions.md](references/source-extensions.md)，选择 baseline 或 experimental 算法。`ariadne-env` 配置修改过的 DIALS Python 源码；`scxrd-env` 保留原始 DIALS。新算法需显式选择，未获实测证据时不得声称提高准确率或速度。MCP 未连接时通过启动器和现有 CLI 执行同一工作流。

涉及多晶格假设时读取 [probabilistic-multilattice.md](references/probabilistic-multilattice.md)。用 `propose_hypothesis` 保存模型与 parent/evidence，不覆盖旧解；用 `start_analysis` 调用软几何匹配、唯一锚点联合精修、局部轮廓、重叠图与像素拟合。匹配权重不强制和为 1，不当作强度占比或孪晶体积分数。反复解释同一质心不等于多次独立测量。

精修的扫描块留出只评估 refinement，不能冒充候选生成也独立留出的盲测。多模型比较须使用同一观察表、标定与固定留出行；训练残差降低不能自动接受。MVP 固定共同仪器几何，多解释的点进入像素联合模型。没有独立锚点时保留未确定状态，不靠复制重叠质心来增加信息。

启用位置扰动积分、Poisson–Gaussian 与边界区间、单源质心 EM 时，先读取 [algorithm-upgrades.md](references/algorithm-upgrades.md)。EM 必须有单源/复合质心选择证据，复合点进入共同像素模型；边界区间不能直接转为 HKLF5 的 Gaussian sigma。

## 先确定数据层级和后端

读取 [workflow.md](references/workflow.md) 与所选路线的 [backends.md](references/backends.md)。先清点用户指定输入，创建全新工作区，保存源文件清单。只复制/解包到新目录，不修改原件权限或原始目录。

```bash
python3 scripts/workflow.py init INPUT JOBS_ROOT --sample sample-name
```

脚本路径均相对本 skill 根目录；实际命令使用绝对脚本路径。返回工作区路径后，所有衍生文件写入该工作区。

- 完整连续衍射帧 + 几何/扫描元数据：允许重新指标化与积分；清点本身不证明完整性。
- 未合并积分观测 + 几何/批次/方向余弦：可按后端能力重新缩放，不能声称重新积分。
- 已校正 HKL：直接交下游解析；不得声称完成原始数据重处理。
- 只有 CIF/RES：仅诊断，不能产生新的实验强度。

不能凭扩展名判定 Bruker `.raw` 是图像，也不能默认 Rigaku `.img/.h5` 可读。SFRM 校准/背景帧另列，保留 matrix/fast 等有用测量，不按文件名一律删除。归档只在新目录安全解包。

按用户要求优先开源 DIALS/cctbx/dxtbx，使用 FabIO/Gemmi 辅助。首先读取 `SCXRD_RUNTIME_CONFIG` 指向的本地环境配置（若设置）；也可发现现有 `scxrd-env` 启动器。未激活 DIALS 时通过该启动器执行，避免改变全局 Python。只有 dxtbx 可读且几何可信，才能继续原始数据处理。商业后端 Bruker APEX/CELL_NOW/SAINT + SADABS/TWINABS 或 Rigaku CrysAlisPro/实际套件是目标确有必要且用户可用时的兜底。先核实版本和参数，不发明厂商 CLI、不购买软件。已获环境安装授权时可下载配置适用开源工具。GUI 是有帮助时的可选路径。

## 执行状态机

顺序为 `intake → import → harvest → index → domains → integrate → correct → export`。每阶段保存输入、命令、版本、输出和决策证据；命令退出成功不等于科学接受。用 `workflow.py run` 保存新 attempt，用 `workflow.py accept` 接受证据；详见 workflow reference。失败时提出可检验假设，只改相关参数，每阶段默认最多三个 attempt，预算可由用户调整。

1. **Import**：确认波长、探测器几何、像素大小、增益/过载、测角仪轴与符号、扫描范围/连续性、温度和标定来源。缺失关键几何时先恢复元数据，不能猜仪器型号。
2. **Harvest**：先保存全部寻峰结果和帧质量概览，再产生专供指标化的初馏子集与逐点剔除原因。冰环、低角区、SNR/帧宽筛选需证据，默认不删单帧/弱点、不设固定分辨率、不开 kNN 密度剪枝。保留未筛选分支核对小晶畴/卫星点。初馏不自动成为最终积分掩膜。
3. **Index**：先无强制高对称性的指标化；依据几何、强度相关性、RMSD、约化晶胞、未指标化点与消光比较候选。记录每次基变换，并同步矩阵和 HKL。晶胞几何不能独自确定空间群。必要时校准光束中心；旋转轴反向需要元数据依据。
4. **Domains**：读取 [twins.md](references/twins.md)。多 sweep 不等于多晶畴，低指标化率不自动等于孪晶。区分同相不同取向、非面网孪晶、面网/赝面网/反演孪晶、异相集合体、调制和无序。保留剩余点。多晶格搜索须比较每一晶畴的独立证据。对未知多相/多晶集合体，读取 [domain-finder.md](references/domain-finder.md)，可用本包的独立 DAFi-inspired 模块生成候选，逐组进入 DIALS 指标化与三维索引残差核验。它是未测试的实验实现，不等同官方 DAFi；候选晶列组不能直接当已接受晶畴。
5. **Integrate**：先静态精修；仅在连续漂移证据和覆盖足够时考虑 scan-varying，跳变分段。比较预测/观测质心、轮廓、背景、过载、峰宽和跨帧截断。涉及重叠时读取 [overlap-integration.md](references/overlap-integration.md)：DIALS 有原生 detector-space deconvolution；本版另实现从原始像素联合拟合、背景消除、完整协方差与可辨识总强度。不能以指标化或文件格式转换代替这些观测处理。
6. **Correct**：检查缩放、吸收、衰减、按扫描角/晶畴/壳层统计，保留校正前后未合并数据、Friedel 对与拒绝记录。经验球谐/多扫描和面指数解析校正按可用证据选择；复合观测可先建几何侧表再运行全局共享尺度/吸收拟合，须核对 rank 与残差分布。不能默认 DIALS 等同 TWINABS，也不重复校正已校正数据。高 outlier 比例提示回看模型，不套用固定禁止阈值。分辨率以多项统计与信息量判断，不为降低 Rint 删弱点。
7. **Export**：写同名 `.ins/.hkl` 和完整还原报告，保留 CELL/基变换/校正记录。未知化学式、Z、cell esd 不得编造；可先用明确标注的解析初始元素/UNIT 占位，必须下游替换，不能作为最终组成。优先厂商或 DIALS 原生导出，避免手拼/截断对称操作。

## 辅助算子

仅在运行任务需要时调用。DIALS 算子在对应环境使用 `dials.python`。

```bash
dials.python scripts/dials_tools.py harvest imported.expt strong.refl harvest-dir
dials.python scripts/dials_tools.py summary indexed.expt indexed.refl domains.json
dials.python scripts/dials_tools.py plot indexed.expt indexed.refl diagnostics-dir --crystal 0
dials.python scripts/dials_tools.py scale-audit scaled.refl scale-audit.json
```

`plot` 是点云与分数指数切片，不是像素重构 precession，不证明消光或漫散射缺失。需要真实倒易层面图时使用原始像素重构/厂商导出。`summary` 报 experiment-to-crystal 映射，不能将 reflection `id` 直接当晶畴号。

未知晶胞多相分域可用 `dials_tools.py q-export → domain_finder.py → dials_tools.py split-candidates → 每组 dials.index → dials_tools.py validate-domain`。完整命令和预算见 domain-finder reference。所有原点和多归属点保留，不能从坐标聚类直接生成重叠总强度。

0.3.0 的候选不提前剥点；`domain_manager.py` 依据三维输入分母、覆盖和残差接受模型，再对原点联合归属。失败候选保留在下一轮搜索。`search_domains.py` 可执行有预算的多轮/多方法搜索，`--heavy` 增加搜索层级；必须明确时间和索引尝试预算，不能无界重试。已有质心方差时 `q-export --with-covariance` 可传播到 q；缺少方差不伪造实测协方差。

像素路线为 `learn_profile_library → overlap_prepare → overlap_integrator → correct_overlap_groups → export_hklf5`。全预测、轮廓校准、像素噪声与真实缩放/吸收因子必须有来源。可用 `--local-grid 3 3 4` 学习有限复杂度的局部模板；必须比较参考支持与 pooled 分支，不能把更密的网格当作更好的模型。完全重合的分量不可辨识时，只输出可测总量；不靠加算力/正则化虚构强度分割。

只有真实观测组满足 [twins.md](references/twins.md) 的契约，才运行 `export_hklf5.py`。该脚本不负责积分、去重叠或吸收校正。

NNLS/Poisson 可用 `overlap_integrator.py --estimator nnls|poisson` 显式选择；保留 `signed-wls` 基线，不能把弱反射的负估计全部裁为零。Poisson 必须有真实计数、增益/偏置与零读出噪声证据；非负边界需非高斯区间，当前实现尚无该区间，相关结果带 quality flag。校正配置需声明 `partiality_policy=already_full_profile_amplitude`，禁止重复 partiality 校正；HKLF5 另需 `phase_relation=same_phase_supported` 与相证据。保留完整协方差 JSONL 和导出行映射，HKL 不是内部主数据格式。

## 衔接解析

读取 [handoff.md](references/handoff.md)。创建交接目录（保留所有上游日志）：

```bash
python3 scripts/workflow.py handoff JOB --solution-ins SOLVE.ins --solution-hkl SOLVE.hkl
```

有非面网孪晶时另传 `--refinement-ins TWIN.ins --refinement-hkl TWIN.hkl`。HKLF 5 本身不当成 SHELXT 解析数据。缺可信解析强度时输出阻碍与中间结果，不伪造 HKLF 4。进入下游后保留分域、吸收、基变换和化学不确定性；下游提出孪晶/高对称性疑问时按证据回退上游，建立新工作区分支。

完整解析授权范围内，交接接受后直接读取可用的 `solve-single-crystal` 并继续其求解/精修流程。找不到该技能或相应软件时，保留可用交接包并说明缺少的依赖。用户要求不测试时，只编写/打包，不运行脚本、示例任务或独立评审测试。禁止执行 SHA-256 校验；现有下游要求此操作时当前用户规则优先。

设计依据、附件保留项与纠正项见 [sources.md](references/sources.md)。
