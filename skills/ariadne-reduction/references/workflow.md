# 执行状态与反馈

## 工作区

`workflow.py init` 接受目录、单文件或 ZIP，创建 `sample_UTCtimestamp_random/`，包含 `input/`、`attempts/`、`evidence/`、`handoff/`（交接时创建）和 `state.json`。原始目录不 chmod、不覆写。ZIP 禁止路径穿越/符号链接；来源清单仅保存路径、大小、mtime 和 SFRM TYPE，不计算 SHA-256。输入完整性仍需导入软件逐帧确认，特别是 HDF5 master 外部链接和扫描断档。

清点扩展名只是候选识别，不是探测器读入承诺。Bruker `.raw` 应结合文件内容区分积分观测和其他数据；SFRM 的 TYPE 为 UNKNOWN 时仍是未确认候选。温度、晶体尺寸等缺失值保留 null。

## 命令与接受

脚本管理顺序：`intake/import/harvest/index/domains/integrate/correct/export`。

命令输入 JSON 是 argv 数组，输出路径相对新的 attempt 目录。用绝对路径引用输入。例子中的文件是用户实际任务时创建的配置，不是此交付内的测试。

```json
{
  "argv": ["dials.import", "/absolute/job/input/run_001.cbf", "output.experiments=imported.expt"],
  "expected_outputs": ["imported.expt"],
  "software_version": "record installed version here",
  "purpose": "Import the selected rotation sequence with its recorded geometry"
}
```

```bash
python3 scripts/workflow.py run JOB import command.json
python3 scripts/workflow.py accept JOB import evidence.json
python3 scripts/workflow.py status JOB
```

接受证据必须包含真实产物路径（相对 JOB）、结论、理由和来源。`accept` 不执行晶体学检验，AI/操作者必须先检查对应数据。

```json
{
  "decision": "accept",
  "reviewed_by": "agent-or-operator",
  "reason": "Measured geometry and frame continuity were inspected; list actual findings here",
  "artifacts": ["attempts/import/01/imported.expt", "attempts/import/01/stdout.log"],
  "metrics": {},
  "limitations": []
}
```

脚本保留命令非零退出、丢失输出和超时状态。新 attempt 不覆写先前结果；同一阶段最多三次，可初始化时 `--max-attempts` 改预算。不自动猜测失败原因，不盲目重试相同命令。阶段接受后锁定；若下游推翻上游，另建版本分支，复制输入并记录父 JOB 和决策，不在已接受阶段原位重跑。

## 各阶段的证据

| 阶段 | 需要检查的内容 | 可接受的产物 |
|---|---|---|
| intake | 数据层级、采集厂商/探测器、关联元数据、完整性线索 | inventory、原始文件列表、分帧/批次计划 |
| import | 实际解码、波长、距离、扫描角、测角仪模型、增益、过载 | experiment/仪器项目与导入日志 |
| harvest | 全寻峰表、候选/剔除点、真实图像、弱点/少数晶畴保留 | 全表、初馏表、逐点原因、mask 来源 |
| index | 约化晶胞/基变换、RMSD、各 sweep 未指标化残差 | 全候选、选定模型、换基操作与图 |
| domains | 晶畴与实验映射、共相性、未指标化点、重叠风险 | 单晶/孪晶/多相分类、矩阵约定 |
| integrate | 轮廓/背景、逐帧漂移、跳变、过载与重叠总观测 | 未合并强度、预测与观测信息、拒绝清单 |
| correct | 吸收/尺度/衰减、误差模型、Friedel 保留、壳层/角度统计 | corrected observations、模型与前后对照 |
| export | 同名 INS/HKL、晶胞/索引一致、格式、真实组成或占位 | solution pair、可选 refinement pair、报告 |

强度统计分别报告 overall、壳层、方向、扫描、晶畴。参考值如 indexing fraction 70%、outliers 2.5%、CC1/2 0.3 只作筛查提示，不是默认硬门槛。没有样本量的均值不证明消光。缺少斑点不证明不存在晶畴。

## 自适应处理

- 寻峰少：先看原图、阈值、噪声/增益和束挡；合理降低阈值并核对真假点，保留弱超结构点。
- 多个 sweep 无法联合：核对轴、定角、detector model 和跳晶；有据再分段，不能从“联合失败”直接反转旋转轴。
- 大胞：结合未指标化周期、低角真实反射和索引候选调 max_cell，不能用近邻距离倒数当最大晶胞轴。
- 高镶嵌/细切片：用真实 rocking profile 拟合宽度；bbox 跨帧数不是 FWHM，profile sigma_m 也不能直接设为猜测 FWHM。
- 分辨率：综合 completeness、multiplicity、I/sigma、CC1/2、方向性和模型信息增益。保留未截断结果；最终 cutoff 写入报告。
- 晶胞连续漂移与溶剂变化：scan-varying 是拟合选项，不是材料呼吸已被证明。
- 多相：每相独立积分/校正/交接，保留共享图像中的重叠污染信息，不能合并成同一个孪晶结构。
