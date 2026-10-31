# 自校准斑点检测与约束轮廓拟合

日期：2026年10月31日。GitHub 基线为 `c7f03742c736b4fb528b19c203ea49a9bb75c8ac`，对应 DIALS 3.30.1。以下是当前工作区的实际测量；分发目录中的旧 wheel 尚未重新打包。

## 数学目标与实现

输入使用 ADU，起始光子量为 `x=(ADU-offset)/gain`。背景采用径向剖面与空间调制的乘积 `b=P(radial_bin) S(y,x)`，在有效的 2×2 像素块上交替更新 Poisson 比值估计，保留原始观测不变。初始空间调制取块内中位数，第二轮校准复用空间调制。

背景残差按背景强度分箱，拟合 `var(x-b)=αb+β`。更新有效增益、基线与读出方差后，在背景像素上估计 12 种位移的相关系数。对可分离核 h，以核自相关 a 计算

```
κ_h = 1 + Σ_d 2 ρ_d a(dy) a(dx) / a(0)²
κ_inf = 1 + 2 Σ_d ρ_d
N = Σ h (x-b)
V = max(b+r, 1e-6) φ [κ_h Q + κ_inf A²/n_eff]
A = Σ_valid h, Q = Σ_valid h²
```

局部背景乘数与超额方差在 2×2 像素块上迭代估计，默认窗口为 9×9 块。验证阶段曾比较 4×4 分块，其较粗的异常裁剪增加了真实帧中未获参考支持的组件，因此保留细分块。

两个尺度分别以二维 Gaussian 场的局部最大数近似确定阈值，并使用 Cornish–Fisher 修正。`false_alarms=0.02` 是这一模型的设计目标，不能当成真实探测器上的已验证误检概率。最终由局部极大、watershed 和尖峰检验确定候选区域。掩膜无效处的分数为负无穷。

C++ 实现保留双精度背景、校准与分数计算。A/Q 核系数为精确二进制分数，少量掩膜变化局部更新；细尺度用五行缓存，粗尺度仅在需要的候选附近计算。背景校准中间阶段只物化所需采样点，最终才生成完整背景；背景插值和裁剪、双通道滑窗分别融合，工作区按调用线程复用。方差拟合按 24 箱重排残差，各箱仍执行完整四轮裁剪，NEON 使用 double 算术。未启用 fast-math，未减少采样、迭代或校准。

macOS 每次调用最多四个工作单元，并发调用共享三个附加 worker 预算；这不意味着整个进程只有四个线程。小于 262,144 像素的图像及其他平台使用单线程。`ARIADNE_MATCHED_THREADS=1..4` 可控制请求数量。工作区按线程隔离，GIL 在内核期间释放。

## 最终合成留出结果

验证集种子 200–219，最终测试集种子 300–319。测试集没有参与局部参数选择。每套 20 张 512×512 图像，包含已知斑点真值、读出噪声、增益、基线、掩膜，以及复杂场景中的环、间隙和尖峰。使用一对一位置匹配，定位门限 3 像素。

表中双方均剔除小于 6 像素的组件。JSON 还保留不剔除小组件的结果。DIALS 的 oracle-noise 对照获得由真实背景和读出方差构造的方差增益图 `1+r/b`，是比忽略读出噪声更强的对照；它不代表 DIALS 官方支持精确 Poisson–Gaussian 似然。

| 最终测试场景 | 方法 | 精确率 | 召回率 | F1 |
|---|---|---:|---:|---:|
| 规则斑点 | Ariadne matched | 0.9985 | 0.6929 | 0.8181 |
| 规则斑点 | DIALS Extended，真实噪声方差图 | 1.0000 | 0.3633 | 0.5329 |
| 复杂探测器 | Ariadne matched | 0.9893 | 0.6388 | 0.7763 |
| 复杂探测器 | DIALS Extended，真实噪声方差图 | 0.9969 | 0.4401 | 0.6106 |

原始结果：[验证集](matched_validation.json)、[最终测试集](matched_test.json)。Ariadne 的 F1 与召回率提高，精确率略低于上述强基线，不能表述为所有质量指标同时占优。

## 真实帧及读取差异

五个扫描各取 8 帧，共 40 帧，1024×1536 像素。参考来自既有双域积分反射；用计算位置、帧范围和 `I/σ` 做召回代理。参考集不是完整真值，组件附近没有已积分反射不等于误检，附近有反射也不是独立真实性证明。

| 同一 underflow 解释、相同 6 像素组件下限 | I/σ≥3 召回率 | 组件数 | 参考支持比例 |
|---|---:|---:|---:|
| Ariadne matched | 0.9666 | 2834 | 0.8345 |
| DIALS Extended，原 panel gain | 0.6520 | 1314 | 0.9079 |
| DIALS Extended，仅有效增益校准 | 0.8046 | 1799 | 0.8994 |
| DIALS Extended，增益与基线校准 | 0.8475 | 1978 | 0.8888 |

DIALS 校准对照使用同一帧背景估出的增益与基线，不借助参考反射选择参数。该对照的检测耗时不包含获得校准参数的 Ariadne 开销；这是偏有利于 DIALS 的时间口径，需与纯内核时间分开阅读。

Bruker FORMAT 100 的 underflow 表按绝对 ADU 解码；[FabIO 官方实现](https://github.com/silx-kit/fabio/blob/main/src/fabio/bruker100image.py) 只对非 underflow 像素加 baseline。安装的 dxtbx 先放入 underflow 值，再对全图加 baseline。实测第一扫描第 20 帧有 82,397 个像素（5.24%）因此增加 64 ADU，涉及真实值 47–64 ADU。另保留原 dxtbx 解释下的完整对照：Ariadne 召回率 0.9714，增益及基线校准 DIALS 为 0.8618；Ariadne 的参考支持比例降至 0.7692。

原始结果：[绝对 underflow](matched_real_absolute.json)、[原 dxtbx](matched_real_dxtbx.json)。这些是检测阶段对照，不能据此宣称完整结构解析的准确率和速度超过 DIALS。

## 数据还原、积分与结构解析入口

agent 与旧 MCP 的 DIALS 命令统一通过 `skills/ariadne-reduction/scripts/dials_runtime.py` 在科学 Python 中运行实际部署 dispatcher。进程内替换 Bruker100 读取：signed i1 underflow 恢复绝对 ADU，baseline 只加到非 underflow 像素，float64 同时保留负 underflow 与 uint32 overflow；不改安装的 dxtbx 文件。新 import 的有效范围覆盖 signed i1，旧 EXPT 的既有范围保持原样。非 Python dispatcher 给出明确部署错误。

阈值回调收到 DIALS 的 `(raw-pedestal)/correction_gain`，`corrected_to_adu()` 根据 panel 常量或 external lookup map 临时还原 ADU，避免再次错误除 gain；也覆盖 ROI。检测的自校准只影响候选选择，不回写 detector gain 或修改积分像素。PixelList、质心和积分继续使用 DIALS corrected 单位。轮廓学习与重叠区域提取等读取原始像素的分析入口也经过同一读取器，包括旧 MCP 的 `start_analysis`。

通过真实生产 backend、Store 和 Context，已对第一扫描的 **32 帧**实际执行：

| 原始帧阶段 | 执行耗时 | 输出 |
|---|---:|---|
| import | 4.59 s | 修正读取的 EXPT |
| matched find_spots | 8.87 s | 389 个斑点 |
| index | 10.48 s | 254/389 分配到一个晶格 |
| refine | 8.09 s | 原生 DIALS 精修输出 |
| integrate | 8.43 s | 1,048 行，1,035 行 sum 标记、971 行 profile 标记 |

新积分结果保留 144 个负 sum 强度。首帧 raw 与 FabIO 逐像素相同，corrected 还原后的检测 mask 与 raw 相同。这里的阶段耗时包含进程启动、I/O 和原生工作量，没有对应整链 DIALS 基线，不能用作加速比。原始数据为双域，这个有限扫描单晶路径只验证软件衔接，没有完成全扫描的多域科学验收。

另以**既有完整积分结果**的原 crystal 0（8,823 行、1,188 个负 sum 强度）验证解析入口。组成 `C20 H16 Co N2 O5` 来自现有 CIF。SHELX export 保留 1,092 个负 HKL 行；实际 SHELXT 生成 `job_a/b/c` 的 RES 和各自 HKL。候选 R1 为 0.270/0.240/0.249，组成和空间群仍需判断；生成文件并不表示结构已解好。这条导出/解析证据使用旧积分结果，不能与上述 32 帧连接后声称新 raw→完整结构成功。

[真实执行报告](reduction_path_run2/reduction_path.json)保存命令、产物、检查与范围；[首次失败记录](reduction_path_run/reduction_path.json)保留临时 overlay 未包含扩展注册元数据导致的参数错误。正常安装已在 `python/pyproject.toml` 注册 entry point；验证覆盖层补入相同元数据后运行成功，生产未增加动态注册分支。完整多扫描约化、缩放、新 raw 路径的全数据结构解析及多域重叠/吸收验收尚未完成。

## 检测耗时与未达到的目标

最终冻结内核对四张 1024×1536 真实帧交替运行 14 轮，剔除各方法前两次样本；单线程和四工作单元顺序逐轮互换，随后运行 DIALS。排除磁盘读取和径向 bins 生成，保留 Python 调用与动态掩膜成本。计时用的临时旧读取器与最终 signed 读取器在这四张图上逐像素一致。

| 同轮方法 | 中位耗时 |
|---|---:|
| Ariadne matched，单线程 | 163.51 ms |
| Ariadne matched，四工作单元 | 87.20 ms |
| DIALS Extended，复用 C++/flex 输入 | 27.95 ms |
| DIALS Extended，包含 NumPy/flex 转换 | 32.63 ms |

四工作单元比同一内核的单线程快 1.88 倍，但仍比 DIALS C++ 慢 3.12 倍。DIALS 执行局部离散统计；本实现额外估计径向/空间背景、两轮有效噪声校准、相关噪声、局部修正和 watershed。这解释了不同的工作量，不能据此宣称速度目标已经达成。四工作单元的各阶段中位数为载入 4.25、背景 18.38、校准 24.28、相关噪声及局部统计 16.18、分数 13.37、区域 4.45 ms；各阶段中位数相加不必等于总耗时中位数。主要开销仍在背景与校准。

测量时机器存在持续外部 CPU 负载。不能把之前较空闲时的绝对时间直接用于计算本轮加速比。原始样本及全部阶段时间见 [最终交替计时](matched_timing_final.json)；[此前交替单线程记录](matched_timing_before_fusion.json)保留为历史数据。下面的 `--compare-threads` 命令能在同一进程中比较两种线程配置，并逐轮反转全部方法顺序，减少固定顺序偏差。

## Gaussian 约束轮廓拟合

Gaussian NNLS 的目标为 `||Ax-b||²`。对增广矩阵 `[A,b]` 做一次 Householder QR，保留最多 K+1 行，再复用这个小矩阵求非负强度、固定分量区间和总强度区间。增广目标列保留参数无关残差；不做秩截断，不形成正规方程，弱方向和重复轮廓保持原有含义。

完整拟合及全部区间的 5 次交替计时中，12,000 行由 50.894 ms 降至 14.041 ms（3.62 倍），60,000 行由 212.244 ms 降至 18.597 ms（11.41 倍）。64 行保留原路径，计时 12.630/12.743 ms，未获得加速。25 对结果的最大区间端点差为 `1.09e-11`，强度差 `3.46e-11`，目标差 `1.46e-11`。

该收益限于固定方差 Gaussian NNLS；没有将其表述为 Poisson/PG 或真实观测组的性能结论。原始数据及每次像素空间 KKT 见 [轮廓压缩测量](gaussian_profile_compression.json)。

## 验证与复现

最终冻结内核的针对性回归为 **90 项通过、45 个子测试通过**，另有两个 MCP 接线测试因科学 Python 未安装 `mcp` 而在主回归中跳过，已在带 `mcp` 的 Ariadne 运行时下单独运行并通过（2 项通过），见[实际回归日志](matched_regression_output.txt)。三条 SWIG 类型弃用警告来自既有 DIALS 绑定。覆盖新旧检测器与 NumPy 输出、掩膜缓存、并行及并发调用、校准统计、ADU/gain/pedestal 单位衔接、signed underflow/uint32 overflow、agent 与旧 MCP 命令接线、非 Python dispatcher 拒绝、Gaussian 原像素目标、梯度、KKT 和区间端点。

独立运行六组合成数据和四张真实帧，单线程及四工作单元共 20 组输出全部布尔像素差为零。噪声统计保持 float64；最大绝对差为读出方差的 `2.28e-11`，来自求和次序，完整误差见 [逐像素和噪声模型对照](matched_parity.json)。这些有限案例证明实际测量的一致性，不代表任意阈值附近输入的位级浮点恒等。

补丁对安装的官方 DIALS 3.30.1 两个未修改源文件执行 `git apply --check --unidiff-zero`，并使用 BSD `patch --dry-run` 复核，均无 fuzz 或偏移；实际应用结果与生产运行所用覆盖层逐字节一致，三个文件均通过语法解析。补丁包含零上下文 hunk，必须按精确行号应用；`git apply` 需 `--unidiff-zero`，否则会直接拒绝。PHIL 参数与扩展实际调用也已检查。

设置 `PYTHONPATH` 包含 `python/src`、`benchmarks` 和 `skills/ariadne-reduction/scripts`，在匹配的 DIALS 环境中运行：

```bash
python benchmarks/matched_spots.py --split validation --output reports/current_validation/matched_validation.json
```

```bash
python benchmarks/matched_spots.py --split test --output reports/current_validation/matched_test.json
```

```bash
python benchmarks/matched_spots.py --work /path/to/ariadne_work --frames-per-scan 8 --reader absolute_underflows --output reports/current_validation/matched_real_absolute.json
```

```bash
python benchmarks/matched_spots_parity.py --experiments /path/to/integrated.expt --output reports/current_validation/matched_parity.json
```

```bash
python benchmarks/matched_spots_timing.py --experiments /path/to/integrated.expt --compare-threads --repeat 20 --output reports/current_validation/matched_timing_reproduced.json
```

```bash
python benchmarks/gaussian_profile_compression.py
```

```bash
python -I tests_rebuild/validate_reduction_path.py --source-expt /path/to/integrated.expt --source-refl /path/to/integrated.refl --composition-cif /path/to/compound.cif --output /path/to/new_run
```

真实链路复现需要已安装 Ariadne 的运行环境；正常的 `sh scripts/install_local.sh` 会通过 `pip --target` 注册 `pyproject.toml` 中的 `dials.spotfinder.threshold` entry point。若仅以只读覆盖层开发，可用 `ARIADNE_VALIDATION_OVERLAY` 指向同时包含补丁后 DIALS 源码和同内容 `dist-info/entry_points.txt` 的目录；该文件只是正常安装元数据的等价物，生产代码不含动态注册分支。逐项检查见[针对性回归日志](matched_regression_output.txt)，原生构建见[构建日志](matched_native_build_output.txt)。
