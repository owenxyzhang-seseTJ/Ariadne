# DAFi-inspired 倒易点分域 · experimental

本模块独立编写，按本仓库许可提供源码。参考 DAFi 的公开算法思想，不复制原厂程序或私有格式实现，不称为官方 DAFi。用户要求暂不测试：当前没有性能、准确率、与 DAFi 等效性或真实衍射数据的验证结果。

DAFi 的任务是从混合点云找出可独立指标化的子集，不是像素积分。本实现也维持该边界。

## 输入输出

输入 CSV：`source_row,q_x,q_y,q_z`，可有额外列。q 为共同参考系中的倒易坐标，单位 Å⁻¹，不含 2π。不能把 detector x/y 和 rotation angle 直接当 q。先导入完整几何，再从 DIALS 实验模型计算。

输出：

- `domains.json`：候选组、方向/周期、参数、搜索预算、归属与限制；A/cell 初始为 null。
- `candidate_NNN.csv`：各候选组的原始点坐标与 source_row。
- `assignments.csv`：每个原始观测属于零个/一个/多个候选；多归属明确标为 ambiguous。
- DIALS bridge 生成 `candidate_NNN.refl` 和 `unassigned.refl`；不把候选表覆盖到全寻峰表。

## 运行关系

实际处理任务中使用以下命令；本次仅编写，没有执行：

```bash
scxrd-env python scripts/dials_tools.py q-export imported.expt strong.refl spots.csv
scxrd-env python scripts/domain_finder.py spots.csv candidates --method hybrid --restarts 3
scxrd-env python scripts/dials_tools.py split-candidates imported.expt strong.refl candidates/domains.json candidate-refl
```

脚本路径需替换成本 skill 内绝对路径。每个候选在全新 attempt 目录做独立索引，不覆盖其他组：

```bash
scxrd-env dials.index /absolute/imported.expt /absolute/candidate_001.refl joint_indexing=True max_lattices=1
scxrd-env python /absolute/scripts/dials_tools.py validate-domain indexed.expt indexed.refl index-check.json --candidate-refl /absolute/candidate_001.refl
```

同一晶体的 sweeps 可联合；不同晶体/参考几何须分别处理。候选独立索引失败或仍有多个模型，回看原始点、方向和单位，不强行认定一个域。后续 `refine_bravais_settings/refine` 仍按正常还原的指标化质量门进行；保存每组 A、约化晶胞、RMSD、唯一 HKL 覆盖、三方向支持和全部残余点。`validate-domain` 只是诊断统计，不能自动证明物相身份/孪晶法则。

## 实现选择

1. 归一化坐标用于方向和晶列搜索，保存逆变换。输出始终恢复原 q，方向/周期字段带 normalized 标识。
2. naive 模式以受限点对抽样找高频方向；smart 模式采样中心点，寻找受多个整数间距支持的共线候选方向。
3. 各方向上把点投影到垂直平面，聚类成晶列；相邻距离和模周期一致性形成候选组。投影聚类使用锚点范围，避免链式近邻把不同晶列并为一列。
4. 候选搜索不剥点。三维核验通过后用 `domain_manager.py` 接受模型并联合归属；仅接受模型下的独占点可从后续搜索种子池排除。歧义/未归属点和失败候选都保留，几何重合不自动生成积分观测组。
5. 搜索时去除重复节点的计数膨胀，输出仍保存全部原始观测。固定 seed 只用于重现抽样，不表明结果确定正确。

这些是本实现的算法设计，不是 DAFi 原程序性能描述。以 DIALS 三维指标化结果接受晶畴；三方向 rank=3 只是必要线索，需要同时检查残差、几何覆盖、候选稳定性、换基和原图。

## 参数与限制

默认 `--minimum-row 4 --min-group 35 --max-groups 8` 是初始启发式，少数晶畴/稀疏扫描须有据调整。`direction-epsilon/row-tolerance/spacing-tolerance` 在归一化坐标下定义，不能直接当实验绝对误差。`max-centres/max-pairs/max-search-points/max-directions` 是计算预算，有限预算可能漏掉弱晶畴；没有找到不证明其不存在。

相同晶列方向/周期的不同相可能暂时进入同一候选组。系统消光会使候选周期为真实周期的倍数，分数周期试探也可能产生超胞/子胞别名；不可直接据此报告最终晶胞。无序/粉末环/动态几何也会污染候选。真正的多相和孪晶分类在独立指标化后进行。

本版不读 `.tabbin/.p4p` 的厂商二进制布局，不覆写输入，不代替 HKLF 5 联合积分。第一轮真实验证应使用用户授权的实测点云，并与独立索引/仪器处理结果比较；当前未开展该轮验证。

## 0.3.0 联合归属、回退与高预算路线

```bash
scxrd-env python scripts/domain_manager.py accept-checks accepted.json candidate-1-check.json candidate-2-check.json
scxrd-env python scripts/domain_manager.py assign spots.csv accepted.json assignment
scxrd-env python scripts/domain_finder.py spots.csv next-candidates --accepted-assignments assignment/assignment.json
```

接受阈值是可配置几何门，不是物相身份判定。索引检查使用原 candidate 输入作为分母；rank 基于去中心化的唯一 HKL，避免一个偏移平面被误称三维覆盖。相同晶格的整数幺模换基别名会合并证据，防止同一取向重复接受。

联合归属优先使用实测质心方差传播得到的 q covariance，缺失时用明确的绝对/分数索引容差，输出几何分数而不冒充后验概率。仪器/模型额外误差仍需外部来源。独占标签仅控制搜索种子，不删除实际积分观测。

速度改动包括 NumPy 批量分箱/计数、区间事件批量排序、投影结果缓存和有界点对抽样；分辨率半径/空间象限分层抽样及不同随机种子用于补充少数晶畴候选。取消原来的“小周期 < 四倍固定容差就拒绝”，改为周期相关的模坐标容差，再由三维指标化否决假周期。尚无实测速度/精度收益。

可自动运行整个候选—指标化—接受—残余搜索循环：

```bash
scxrd-env python scripts/search_domains.py spots.csv imported.expt strong.refl search-job
scxrd-env python scripts/search_domains.py spots.csv imported.expt strong.refl heavy-search-job --heavy --max-index-attempts 128 --max-seconds 14400
```

它执行真实 DIALS FFT3D/FFT1D 尝试，保存命令/日志/接受及拒绝记录，固定共同仪器几何以保持 q 基准一致。搜索超时和失败不会剥离候选点。只接受几何模型；同相/孪晶/调制判定仍由完整证据决定。这些命令本次未执行。

参考：[Aslandukov et al., J. Appl. Cryst. (2022), DOI 10.1107/S1600576722008081](https://journals.iucr.org/j/issues/2022/05/00/iu5028/index.html)。
