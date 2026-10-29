# 复合观测的全局尺度与方向吸收

`global_scale.py` 在所有组上联合拟合 `y_g = D_g [ known_forward * exp(X beta) * I_equiv ]`。可辨识分量使用完整分量协方差和单位设计矩阵；仅总量可测的组使用一行求和设计。不预先拆分复合总量，也不复制总强度到各畴。

同一空间群等效类在每个晶畴内共享 I；不同晶畴的 I 分开保存，保留畴间比例。Friedel 是否合并由每个域的 `anomalous` 显式指定，默认不猜。所有晶畴共享 beta：每原始扫描最多二阶角度趋势、相对扫描偏置，以及入射/出射方向的五个实 l=2 偶次谐波和。方向在各畴晶体 Cartesian common setting 下，几何须有来源。吸收是经验相对传输模型，不估计绝对吸收系数或晶体外形，不宣称与 TWINABS 全功能等价。

交替求解：固定 beta 用 WLS 求有符号等效强度；固定 I 求非线性 beta。仅接受 chi-square 不增加的状态，要求梯度、收敛、联合参数秩与预算满足条件。特征整体中心化固定共同尺度规范；没有独立域尺度归一化。参考等效反射或方向覆盖不足时，联合 Jacobian 检出零空间并保留 unresolved，不能用正则项制造信息。

## 输入与入口

先用 `scaling_geometry.py INTEGRATED.jsonl MODEL.expt FULL_PREDICTIONS.refl EVIDENCE.json NEW_GEOMETRY` 建立侧表。prediction_row、experiment_id、HKL 必须逐项匹配。原生 DIALS `compute_corrections` 给出 LP/QE；已知 forward multiplier 为 `qe/lp`，校正时取倒数。入射/出射方向由固定仪器、旋转角和晶体 U 反变换计算。

EVIDENCE.json 包含 basis、geometry_provenance、symmetry_provenance、complete_contributor_audit=true，以及 domains，例如域 1 提供 space_group、anomalous 布尔值、experiment_ids。若要导出孪晶 HKLF5，还须提供 phase_relation=same_phase_supported 和 phase_evidence_provenance。可设 scan_order=0..2、absorption_degree=0 或 2；降低复杂度必须有实际几何和残差依据。

再运行 `global_scale.py INTEGRATED.jsonl NEW_GEOMETRY/scaling_geometry.json NEW_SCALE`。也可提供有实际来源的同schema侧表，不需要读原始图像。MCP start_analysis 的 scaling_geometry 输入顺序同上四个文件；global_scale 输入顺序为积分 JSONL、几何 JSON。

## 输出与不确定性

- scaling.json 保存等效强度、共享 beta、完整联合协方差、参数顺序、每组预测/残差、同源数据敏感度、收敛轨迹、秩、预算及排除记录。它是主结果，未收敛时仍保存诊断，但不生成可用校正表。
- 成功时 corrections.json 给出实际估计的校正因子、因子协方差、强度-因子同源交叉协方差；corrected/corrected_observations.jsonl 使用现有校正器传播这些量。
- HKLF5 的 Gaussian sigma 来自局部联合线性化；输入组独立、轮廓、LP、对称性和几何视为已知。不以 reduced chi-square 重新缩放已知测量协方差。跨组校正相关性保留在主记录，HKLF5 无法编码。
- 总量可测、各分量校正不同的组参与全局拟合，但不会由等效平均预测替代实测分量导出。fit_only_observations 记录这些组。只有同一随机校正能作用于所有分量时才校正原始总量。

## 验证状态与规模限制

已知真值算例覆盖 90 个混合分量/总量观测组、12 个等效类、6 个共享参数，恢复值与真值一致，并走通从 corrections 到 HKLF5 的链路；内核另有独立数值 Jacobian 对照的协方差、同源交叉项和退化/预算检查。真实 DIALS 两晶格几何上，预测行号、HKL、单位方向与 `lp_forward=qe/lp` 逐项核对通过。这些只说明实现与约定一致。

尚未用实测孪晶原始帧验证，未做跨仪器或盲测比较，未估计绝对吸收系数、晶体外形或独立晶畴尺度。内核是稠密矩阵实现，默认上限 4096 个测量行、256 个参数、30 秒；更大数据须按扫描或等效类分块，或提高预算并接受相应内存与耗时。

这条路线没有估计独立 domain multiplier，没有自动离群值删除，也不把强度等效一致性当成同相或孪晶律证据。真实数据使用前必须审查残差随扫描/方向/晶畴的分布，并有足够等效覆盖。
