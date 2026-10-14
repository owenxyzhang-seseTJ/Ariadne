# 晶畴、孪晶与观测组

## 分类与证据

| 现象 | 需要确认 | 路线 |
|---|---|---|
| 多 sweep | 同一个晶体模型还是多个样品/跳变段 | 同晶体可 joint indexing；独立样品分开 |
| 相同约化晶胞、不同 U | 取向关系、真实残余晶格、随角度一致性 | 多晶畴；先诊断，不自动命名为孪晶 |
| 完全重合 | 晶格对称与结构点群关系 | HKLF 4 + 有据 TWIN/BASF 的下游精修 |
| 部分/分裂重叠 | 各 A 矩阵、像素/转角轮廓、贡献域 | 联合积分与分域缩放；HKLF 5 精修 |
| 不同约化晶胞 | 多相证据、可能换基/超胞关系 | 每相独立工作流；不写成同相孪晶 |
| 卫星点/条纹 | 调制、漫散射、孪晶或多相 | 保留原点，交适用调制/重构软件 |

矩阵均写清空间、单位、左右乘和基准。若 `q=A_i h_i=A_j h_j` 且 h 为列向量，则 `h_j=A_j^-1 A_i h_i`。实验室旋转 `U_j U_i^T` 不能直接当 SHELXL TWIN 的整数索引法则；换基后须重算。三轴约化晶胞相近不能证明同相。

## 重叠处理

不同角度的 q 相近不是同一次曝光；同一像素相近也可能跨不同 sweep。判断重叠至少使用 experiment/imageset、panel、scan/frame interval、预测位置/轮廓宽度和 integration group id。

完全重叠观测测得总强度，其 sigma 来自该联合观测。部分重叠须使用后端明确的联合/分离轮廓模型及误差传播。不得将已独立积分的 I_main 复制成“重叠总强度”；不得相加 I_i 却忽略协方差；不得只导出主晶畴而丢掉其他域的独立点。

0.3.0 已加入独立像素级重叠积分及校正交接，见 [overlap-integration.md](overlap-integration.md)。完全重合时只保留可辨识的总观测；分离分量不发布任意伪逆拆分。不同物相不能合并成一个 HKLF 5 孪晶结构。

## HKLF 5 的输入契约

优先接受后端原生 HKLF 5。`export_hklf5.py` 用于后端可输出真实观测组而未写 SHELX 格式的情况。输入为 JSON Lines：首行 metadata，后续每行一个已校正观测。每个贡献者的 hkl 已换到下游模型的共同结构基准；晶畴编号从 1 连续编号。

```json
{"record":"metadata","schema":"scxrd-observations/1","corrected":true,"basis":"explicit downstream structure basis","backend":"actual program/version","integration_provenance":"actual log path","correction_provenance":"actual log path","domain_count":2,"intensity_semantics":"measured_group_total","all_contributors_accounted":true,"excluded_observations_provenance":"actual rejection ledger or explicit none","overlap_model":"actual joint integration model"}
{"record":"observation","observation_id":"sweep1:panel0:frames10-12:group17","intensity":1500.0,"sigma":24.0,"contributors":[{"domain":2,"hkl":[-1,2,0]},{"domain":1,"hkl":[1,2,0]}]}
{"record":"observation","observation_id":"sweep1:panel0:frames16-18:group18","intensity":-3.2,"sigma":5.1,"contributors":[{"domain":2,"hkl":[2,1,0]}]}
```

数值是格式例子，不是测量/测试数据。真实任务必须换成真实值和来源。强度允许负数；sigma 必须有限且 >0；不筛 I/sigma；每组保留一个实际测量。独立点 m 为正；重叠组除最后一个贡献者外 m 全负，最后一个 m 正，强度取最后一行。参考 SHELXL 官方定义。

```bash
python3 scripts/export_hklf5.py corrected-observations.jsonl twin.hkl
```

脚本统一缩放 I 与 sigma 以适应 F8.2，仅转换格式并记录 multiplier；不估 BASF、不构造 INS、不证明积分模型。INS 由原生后端或有据模板生成，HKLF 5 不与 TWIN 同时使用，BASF 的起点与最终精修值须区分。多域 BASF 之和 <1，不能从各域平均强度猜体积分数。

## 解析和精修数据分开

SHELXT 解析使用可信 HKLF 4（后端提供的去孪晶/主域不重叠数据等，记录来源及完整度）；原始总强度 HKLF 5 留给 SHELXL。非重叠子集不足解析时保留阻碍，不重复复制总强度来伪造主域。已有模型可直接进入有据 HKLF 5 精修；本版 `workflow.py handoff` 的自动解析入口仍要求 HKLF 4，已有模型走下游技能的独立精修请求。
