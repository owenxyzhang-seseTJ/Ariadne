# Ariadne 0.5.0 源码扩展与 agent 接口

本版是实验实现。2026年10月26日新增方法已运行针对性数值与 DIALS 合成接口验证，见 [algorithm-upgrades.md](algorithm-upgrades.md)；实测数据完整基准仍未执行。以下记录实现与依据，不提供普遍性能结论。

## 修改范围与部署

固定上游版本 DIALS 3.30.1，下载地址为官方 `dials/dials` 的 v3.30.1 tag。修改两个 Python 核心文件，并新增一个 DIALS threshold extension：

- `src/dials/algorithms/indexing/lattice_search/__init__.py`：新增 PHIL 参数、评分器选择及候选证据输出。
- `src/dials/algorithms/indexing/model_evaluation.py`：显式选择新评分器时划分 refinement training/holdout，固定共同仪器几何，返回精修晶胞和结构化评分。
- `src/dials/extensions/ariadne_spotfinder_threshold_ext.py`：加入可配置的候选像素检测插件。

核心补丁位于 plugin 根目录 `patches/dials-3.30.1-ariadne.patch`；DIALS 原许可保留在 `patches/DIALS-LICENSE`。新 companion 模块和工作流为 MIT。启动器通过显式 PYTHONPATH 加载匹配源码和 companion metadata，复用已安装的 DIALS/cctbx/dxtbx 编译扩展；没有修改或重新编译 C++。这不是经过完整编译验证的独立 DIALS 发行版。

## 寻峰：多尺度统计提议

开启方式：

```bash
ariadne-env dials.find_spots imported.expt \
  spotfinder.threshold.algorithm=ariadne \
  spotfinder.threshold.ariadne.gain_adu_per_photon=CALIBRATED_GAIN \
  spotfinder.threshold.ariadne.noise_model=poisson
```

`CALIBRATED_GAIN` 必须替换为真实标定值，不是可直接运行的示例。CCD/积分型探测器有读出噪声时选择 `poisson_gaussian` 并给出 `readout_variance_adu`；此分支使用 quasi-Gaussian excess approximation，没有声称实现完整 Poisson-Gaussian 卷积似然。多个 panel 当前共用同一配置，噪声不同时应分开配置处理。

mask-aware 环形背景排除最大候选支持，迭代剔除正向背景污染，使用 1/3/5 像素 box-sum 搜索。Poisson 分支计算单侧 deviance `2[n log(n/mu) - n + mu]`；预期计数下限 0.5 是数值正则，不是实测背景。统计只提议候选像素，DIALS 继续三维连通斑点构建与物理指标化。box-sum 的计算复杂度随像素数和尺度数线性增长，实际速度未测。

此处有背景估计、迭代排点和多尺度选择，score 不能当已标定的显著性或假阳性概率。弥散散射、粉末环、卫星、非高斯尾部仍可能被提议。以全流程指标化、假阳性、强度不确定性与最终数据质量评价，不能以点数增加判定优劣。没有引入未经训练的深度学习模型。

## 晶胞：conditional refinement holdout

```bash
ariadne-env dials.index imported.expt strong.refl \
  indexing.basis_vector_combinations.solution_scorer=ariadne \
  indexing.basis_vector_combinations.ariadne.evidence_directory=candidate_evidence
```

默认约 20% 扫描块不参与短程晶体精修；每块默认 20 帧，按原 imageset_id 划分，不能用随候选变化的 assigned id 划分。候选生成、初次 HKL 分配、非原胞修正仍使用全部输入。因此这是**有条件的精修留出**，不是独立端到端交叉验证。各候选使用相同点顺序、seed、几何、噪声假设；beam/detector/goniometer 在新评分分支固定。已有问题几何需要单独标定后再比较。

对留出 q 计算周期 Gaussian/uniform mixture density ratio，显式带 `abs(det(A))` 归一化，降低过密倒易点阵靠偶然吻合获胜的倾向。q_sigma 是配置假设，当前不是实测协方差。周期整数邻居范围依据 `sigma * row_norm(inv(A))` 自适应扩大，默认覆盖 6 sigma、每轴半径预算 6、每批 512 点；大晶胞不会一律受限于固定 27-image 近似。超预算时明确拒绝，要求改善约化基/噪声标定或扩大显式预算。截断仍是数值近似，未验证误差界。保留最低覆盖、三维 HKL rank 以及正的平均 log density ratio gate；不足时明确失败，不默默切换新评分的定义。JSON 记录每个候选的 cell、A、coverage、q residual、周期范围与假设；不推断空间群。

用 `search_domains.py --solution-scorer ariadne` 接入原来的分域搜索。失败候选不剥点、所有已接受模型联合归属、歧义保留。`--heavy` 扩大候选、采样、重启和方法预算；仍需设置时间和次数上限。已知晶胞先验由 DIALS 原有 known_symmetry/known-cell 搜索处理，并记录先验来源。本版没有完整的鲁棒整数最小二乘晶胞恢复器，也没有新增严格多晶格 Bayesian selection。

## 局部峰形与重叠

```bash
ariadne-env python scripts/learn_profile_library.py REFERENCES.refl calibration.json profiles \
  --local-grid 3 3 4 --min-local-references 5 --shrinkage-references 20
```

保留默认 pooled (1×1×1) 分支。局部网格按 x/y/scan 帧划分真实孤立参考，峰形对齐后非负归一。局部模板使用 `alpha=n/(n+lambda)` 向 pooled 模板收缩；共享完整有限支持，推断时使用非负 trilinear 权重。缺参考节点或超出参考范围回退 pooled，不拟合每个弱反射的任意形状。选择后的模板传入已有原始像素联合 QR/SVD 重叠拟合，而非只用于图形诊断。

新增 bounded LRU template cache，避免每个反射重复从磁盘读模板。局部模型按 experiment/domain 与 panel 分离；尚未跨 sweep 池化，也未把探测器空间经验模板变为严格 reciprocal-space physical local model。它不能单独区分仪器漂移与晶体变化。强度协方差仍以已给定峰形、几何、噪声为条件，没有传播参考模板的不确定性。

可辨识性仍优先：可辨识分量给 I 与完整条件协方差；只能测总量就保留 total-only；两者都不可信保留 unresolved。重叠支持/背景图连通后只读取一次共同像素。力大砖飞可增加搜索、参考数据与物理模型比较，不能凭正则化拆出完全相同 profile 的个体 I。

## 动态实验

`geometry_diagnostics.py` 用显式 experiment→domain/raw_scan mapping 输出每 panel/scan-bin 的 robust residual 与跨畴对比。mapping 的 `frame_offset` 必须有来源，不能直接把 sweep id 当 raw_scan。输出 `cause=unresolved`；共同漂移可能来自仪器，也可能来自样品整体运动。

仍采用 DIALS 原有 static/scan-varying physical refinement，必要时按跳变分段。本版没有新增可以从一个数据集唯一分离真实动态结构与所有仪器误差的模型；该目标需要 geometry Jacobian、共同几何约束与独立标定/重复扫描。禁止将描述性残差图表解释为结构变化证明。

## Agent 原生接口

MCP stdio server 使用官方 Python SDK 1.30.0，配置在 plugin 根目录 `.mcp.json`。工具：`capabilities`、`create_job`、`start_operation`、`inspect_job`、`read_artifact`、`record_acceptance`、`make_handoff`。提供结构化输入/输出；长计算立即返回 operation id，后台 worker 保存状态、argv 与日志。只允许固定 DIALS 操作和 typed 参数；输出固定为当前新 attempt 内文件，不接受 shell command。单任务文件锁避免 MCP 并发修改同一个科学状态，超时终止该计算进程组。

Ariadne 0.5.0 另加入 `start_analysis`、`propose_hypothesis` 和事件/假设记录，当前总计 9 个 MCP tools；新的 NNLS/Poisson、软匹配、native joint refinement 和固定留出比较契约见 [probabilistic-multilattice.md](probabilistic-multilattice.md)。

科学接受始终需读取真实 evidence；结果存在、退出码 0、文件格式正确都不足以代替科学判断。MCP native scale/export 对适用的单相路线有效。复合观测的共享全局尺度/吸收由 `global_scale.py` 估计，须先建几何侧表并检查 fit status；不能冒充 TWINABS。

重启后的 active operation 若没有 result，先查 worker.log 与进程状态；不自动重放科学计算。下游 ready 后调用可用 `solve-single-crystal` skill，保留 HKLF4 解结构/HKLF5 精修的区别。

## 原理与实现来源

- [DIALS 原理、数据模型、寻峰、经验峰形与几何精修](https://journals.iucr.org/d/issues/2018/02/00/di5011/) — DOI 10.1107/S2059798317017235。
- [DIALS 官方 extension interface](https://dials.diamond.ac.uk/documentation/library_reference/extensions/index.html)。
- [TakeTwo 与已知晶胞多晶格思路](https://journals.iucr.org/d/issues/2016/08/00/rr5128/) — 本版没有复制其实现，也未宣称该算法已接入。
- [DIALS 3.30.1 上游源码](https://github.com/dials/dials/tree/v3.30.1) — 本版直接阅读 threshold extension、model_evaluation、lattice_search、gaussian_rs sampler 与 native overlap fitter 后修改。
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) 与 [Codex plugin packaging](https://developers.openai.com/plugins/build/plugins)。

后续优先实数据对比：弱斑 false positives/漏检、每畴三维留出误差、孤立峰形留出预测、重叠可辨识强度和 covariance coverage、最终等价反射一致性与结构解析成功。当前用户要求不测试，这些仍未执行。
