# DIALS 源码对照与 matched 内核热点重写

日期：2026年11月02日。修改前源码基线为 `3ed9779ab22bd209b5ad997a41dfa2899de63465`，对照本机 DIALS 3.30.1 的生产 `DispersionExtendedThreshold<double>`。本报告只评价斑点检测内核及其接口，未执行全数据联合孪晶积分或结构精修。

## 结论

在四张 1024×1536 真实帧上，同进程加载修改前后两个原生扩展，与 DIALS 交替运行 26 轮；每轮反转顺序，各方法剔除最初两次样本：

| 方法 | 中位耗时 |
|---|---:|
| 修改前 matched，单工作单元 | 69.3614 ms |
| 修改后 matched，单工作单元 | 47.5617 ms |
| 修改前 matched，四工作单元 | 36.1054 ms |
| 修改后 matched，四工作单元 | 24.9559 ms |
| DIALS Extended，复用 C++/flex 输入 | 11.6393 ms |
| DIALS Extended，包含 NumPy/flex 转换 | 13.8807 ms |

四工作单元延迟降低 **30.9%**，加速 **1.447 倍**；单工作单元延迟降低 **31.4%**，加速 **1.458 倍**。修改后四工作单元仍比 DIALS C++ 慢 **2.144 倍**，尚未达到速度超越目标。此前 87.20 ms 的记录属于另一负载条件，不能据此计算本轮收益。

24 组合成双峰和 40 张真实帧的优化前后检测 mask 全部逐像素一致。另有 6 组合成图和 4 张真实帧分别用一／四工作单元对照 NumPy，共 20 组，像素差均为零。有限测量支持本轮加速未损失既有弱峰与近邻峰分辨能力，不能证明任意阈值附近输入的浮点位级恒等，也不能替代孪晶联合积分的科学验收。

原始计时：[matched_rewrite_timing_final.json](matched_rewrite_timing_final.json)。分辨能力：[matched_resolution_guard.json](matched_resolution_guard.json)。NumPy 对照：[matched_rewrite_parity_final.json](matched_rewrite_parity_final.json)。

## DIALS 为什么更快

必须区分生产类与 Debug 类。生产入口在 `vendor/dials-3.30.1/src/dials/algorithms/image/threshold/local.h:1432`，实际顺序为：

1. `compute_sat` 一次遍历融合有效计数、强度和平方和，写入复用缓冲区。
2. `compute_dispersion_threshold` 通过积分表四角查询直接比较离散度判据。
3. `chebyshev_distance` 正反两遍递推，再一遍转成背景掩膜。
4. 第二次 `compute_sat` 在同一缓冲区累计背景像素。
5. `compute_final_threshold` 查询局部均值并直接写最终 mask。

因此主体约为七次全图遍历，另有缓冲区初始化。它没有沿生产路径物化均值、方差、离散度等全部统计图；这些属于 Debug 类的另一套调用链。积分表将窗口查询由邻域面积复杂度降为每点固定四角读取，距离变换也避免了逐点完整邻域搜索。

无 gain 的离散度判据在 `local.h:1101` 附近直接比较

```
m*y - x*x - x*(m-1) > x*nsig_b*sqrt(2*(m-1))
```

其中 `m、x、y` 分别是有效计数、强度和与平方和。最终使用 `I >= mean + nsig_s*sqrt(mean)`。它通过直接判据节省了中间统计图、除法与后续遍历。第二次积分表仍计算最终阶段不再使用的平方和，因此 DIALS 也并非没有冗余。

matched 的工作职责更多：径向／空间背景初始化与交替拟合、两轮噪声校准、12 个 lag 的相关噪声、三轮局部尺度、两个核尺度和 watershed。把它直接替换成 DIALS 判据会改变模型及弱峰行为。本轮保留全部模型步骤，优化其中重复工作和串行执行。

需要纠正先前“只能改变精度或模型才能继续压缩”的判断：实际测量已经证明，保留模型仍有显著实现优化空间。

## 实际热点与改动

修改前同轮单工作单元的阶段中位数约为背景 19.93 ms、校准 24.27 ms、相关噪声与局部统计 14.44 ms、评分 4.71 ms。额外的背景内部计时表明，`fill_tiles` 在这些真实帧上仅约 0.01–0.03 ms；它不是主要瓶颈。粗网格裁剪、插值、中位数初始化以及校准才是重点。

实现位于 `python/src/ariadne/_native.cpp`：

- 背景四条径向归约链独立调度；每条链内部的像素顺序与跨链合并顺序保留。
- 每次径向剖面更新后缓存 `ns*P[bin]`，裁剪与 tile 分母累计复用相同 double 结果，保留 `(ns*P)*S` 的结合顺序。
- 背景插值、裁剪及最终像素展开改用两路 float64 NEON；tile 求和继续按原来的逐元素顺序累加。保留严格／非严格比较符号和公式括号。
- 缓存最初 `(input-offset)/gain` 的 double 结果，两轮校准仍从它计算 `(x0-O)/G`。没有改成递推校准，也没有合并分母。
- 删除第二轮结束后无人读取的 `bg.e` 校准写回；第一轮仍保留以供 warm start。
- 局部统计复用现有块有效计数，消除复制；将已有倒数表达式按 bin／块只计算一次。
- 相同输入的最后一次 `window_pair(kn,kn)` 改为单通道模板实例，保留原通道滚动求和顺序。
- 局部尺度裁剪和条件除法改用 float64 NEON，保持三轮迭代。
- 12 个 lag 独立调度，各自内部累加顺序保留。
- fine 评分以行段并行，每段携带两行 halo 和私有五行环形缓存；残差只由所属行段写入，halo 不发生共享写入。

未启用 fast-math，未改为 float32，未减少背景轮数、两轮校准、四轮噪声裁剪、三轮局部修正、任何采样点或阈值二分次数；阈值、watershed、峰间对比度、小组件下限及尖峰剔除规则保持原样。并发调用继续使用既有 worker 预算与 thread-local 工作区。

两个 Python ABI 的源码内原生扩展均已构建（3.12、3.13）。`dist/` 的 wheel 未重打包。缓存初始输入增加一张全图 double 数组，移除 `local_ns` 抵消一张粗网格数组；这是用内存换取重复除法的消除。

## 最终阶段耗时及剩余差距

| 四工作单元阶段 | 修改前中位数 | 修改后中位数 |
|---|---:|---:|
| load | 1.915 ms | 1.612 ms |
| background | 7.873 ms | 4.234 ms |
| calibration | 10.045 ms | 6.597 ms |
| noise（含 local scale） | 7.468 ms | 5.890 ms |
| scores（含 mask cache） | 4.726 ms | 2.644 ms |
| basins | 1.723 ms | 1.883 ms |

各阶段中位数之和不必等于总耗时中位数。basins 未修改且本轮中位数略增，不能把所有阶段都描述为加速。背景、校准和局部统计仍是主要开销；完整 matched 路径仍明显慢于本机 DIALS。

计时包含公开 Python 调用、转换与动态 mask 检查，排除磁盘读取和径向 bins 生成。DIALS C++ 项复用 flex 输入，另列转换项。计时只比较这套固定检测配置，没有按双方相同召回率重新调阈值，也不代表整条约化流程吞吐量。

## 弱衍射与近邻双峰验收

新定向基准 `benchmarks/matched_resolution_guard.py` 固定种子 801、802、803，不据结果调参。每张 512×513 图像包含 25 对 Gaussian 峰；宽度 sigma=1.1 像素，间距为 2／4／6／8 像素，强度比 1:1 或 1:4，弱分量总光子数覆盖 40–640。背景有梯度，含 Poisson 和 Gaussian 读出噪声，输入 gain=25、offset=64。

组件与真值采用一对一匹配，位置门限为 `min(1.5, separation/3)` 像素；一个合并组件不能同时算作两个峰。双方统一采用至少 6 像素组件。DIALS 对照额外获得由已知背景和读出方差构造的 oracle-noise 图。

| 峰间距 | 总峰对 | 修改前 matched 分开 | 修改后 matched 分开 | DIALS oracle-noise 分开 |
|---|---:|---:|---:|---:|
| 2 px | 150 | 0 | 0 | 0 |
| 4 px | 150 | 25 | 25 | 6 |
| 6 px | 150 | 117 | 117 | 72 |
| 8 px | 150 | 124 | 124 | 87 |

完整 JSON 还保留每种强度比、种子、SNR 分段召回与 FP。这里的 2 像素结果说明现有检测器存在明确分辨极限。模拟双峰是邻近投影的压力测试，不能等同于完整孪晶衍射模拟。

真实验收使用此前同一五扫描的 40 帧，逐帧比较修改前后 mask，再与既有多晶格积分位置作邻近召回对照。所有 mask 一致，分模型统计也一致。扫描各自具有独立 crystal 模型，共 10 个模型；没有把模型编号擅自当成已确证的全局物理孪晶域。

| 既有积分 I/σ 区间 | 参考数 | 修改前／后 matched 找回数 | DIALS 原 panel gain 找回数 |
|---|---:|---:|---:|
| <1 | 1195 | 31 | 13 |
| 1–2 | 363 | 49 | 1 |
| 2–3 | 220 | 135 | 2 |
| 3–5 | 226 | 199 | 14 |
| 5–10 | 254 | 250 | 132 |
| 10–20 | 239 | 237 | 214 |
| 20–50 | 184 | 182 | 178 |
| ≥50 | 146 | 146 | 146 |

真实参考不是完整真值，弱参考附近检出像素也可能来自另一重叠分量。这里能证明优化未改变这些帧的弱峰选择，不能用邻近召回证明孪晶强度拆分正确，也不能把它称为精确率。全数据多域联合积分、吸收／缩放及孪晶结构精修仍未完成科学验收。

## 数值与接口检查

- 54 项通过：matched 与 NumPy、一／四工作单元、并发、奇数尺寸、非连续输入、mask 缓存、旧检测器和 ADU／gain／pedestal 生产接口。见 [matched_rewrite_tests_final.txt](matched_rewrite_tests_final.txt)。
- 4 项既有进程边界检查通过。首次额外检查因科学 Python 缺少 `mcp` 有 2 项失败，补入已有部署目录后复核通过；未修改相关代码。见 [首次记录](matched_rewrite_tests.txt) 和 [复核记录](matched_rewrite_boundaries_retry.txt)。
- Python 3.13 原生扩展另跑同一 35 项 matched 检查通过，见 [matched_rewrite_tests_py313.txt](matched_rewrite_tests_py313.txt)。不能将不同 ABI 重跑累加成不同测试数。
- 最终 20 组 NumPy 对照全部检测像素差为零，噪声诊断检查 `rtol=1e-7, atol=1e-9`。
- 24 组合成双峰及 40 张真实帧的修改前后 mask 全部一致，最大读出方差绝对差为 `2.28e-11`；其余诊断误差在 JSON 中逐项保留。这是 double 实现间末位数值差的实测范围，不是任意输入的误差上界。

未做 SHA256 校验，未运行与检测优化无关的 smoke test，未修改安装的 DIALS 源文件，未提交或推送。

## 复现

在匹配 DIALS 3.30.1 的科学 Python 中，设置 `PYTHONPATH` 包含 `python/src` 和 `benchmarks`。优化前原生扩展应从上述 git 基线的 `_native.cpp` 单独构建，保存在不会被新构建覆盖的目录；`--baseline-native` 指向该 `.so`。本轮临时旧二进制用于同进程实验，收尾后删除，原始测量 JSON 中仍保留当时输入路径。

```bash
python benchmarks/matched_spots_timing.py --experiments /path/to/integrated.expt --compare-threads --baseline-native /path/to/before/_native.cpython-312-darwin.so --repeat 26 --output reports/current_validation/matched_rewrite_timing_reproduced.json
```

```bash
python benchmarks/matched_resolution_guard.py --baseline-native /path/to/before/_native.cpython-312-darwin.so --work /path/to/ariadne_work --output reports/current_validation/matched_resolution_guard_reproduced.json
```

```bash
python benchmarks/matched_spots_parity.py --experiments /path/to/integrated.expt --output reports/current_validation/matched_rewrite_parity_reproduced.json
```
