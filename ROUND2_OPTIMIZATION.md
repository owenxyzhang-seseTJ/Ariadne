# Ariadne 第二轮：物理导数、统计压缩与结构化求解

本轮基于 **2026-10-29 第一轮交付**继续修改实际源码，重点覆盖反射预测、
仪器几何、体素积分轮廓、有符号 WLS、非负计数似然、条件剖面区间、缩放和导出。
原始 Git 基线仍为 797ac22ffa9b81da2dd977e013bea8f114b49eb2；
原生接口目标保持 DIALS **3.30.1**。

完整推导：[ariadne_round2.tex](reports/ariadne_round2.tex)。
编译稿：[ariadne_round2.pdf](reports/ariadne_round2.pdf)。
第一轮代码、原理报告和实测证据均保留；以下速度比较全部以第一轮为基线。

## 已落地的算法改变

| 环节 | 实际修改 | 数学作用 |
|---|---|---|
| 旋转反射预测 | rotation_geometry.py、centroid_em.py | 对 Ewald 约束及面板透视投影解析求导，保留原生预测器决定的分支、面板、扫描圈数。 |
| 非互斥几何匹配 | soft_multi_indexer.py、centroid_em.py | 每个观测只做一次协方差 Cholesky；完整候选集分块求责任度，省去三维残差张量。 |
| Gaussian 体素 | profile_geometry.py | 相邻体素共享稳定 CDF 边界；位置导数是下表面通量减上表面通量。 |
| 轮廓准备与学习 | overlap_prepare.py、profile_learning.py | 预分配输出；经验插值分块；每个学习参考只对齐一次，保存池化和局部累积量。 |
| 有符号 WLS | orthogonal_compression.py、overlap_integrator.py | 对 [背景, 轮廓, 数据] 做增广分块 QR，在小因子中保留整个二次目标、常数残差及弱可观测模式。 |
| 非负计数拟合 | constrained_pixels.py | 严格零轮廓像素的参数无关因子移出迭代；满足条件的 Poisson 观测压缩为充分统计量。 |
| 条件剖面区间 | constrained_pixels.py | MLE 中心化的逐观测似然差；独立 Poisson 总量剖面转成标量对偶根；一般问题用可行状态续接。 |
| 区间验收 | constrained_pixels.py | 无量纲括区求根；端点必须实际达到指定 LR 阈值，避免强度单位改变精度。 |
| 全局缩放 | variable_projection.py、composite_scaling.py | 固定贡献算子一次组装，按保守支撑图批量线性求解；只优化尺度，使用完整剖面残差导数。 |
| HKLF5 | hklf5_stream.py、export_hklf5.py | 两遍流式验证与写出；保留同组观测语义、负测量及共同尺度；显式检查弱 sigma 的量化可表示性。 |

以上脚本均位于 skills/ariadne-reduction/scripts/。
原有 DIALS 三文件源码补丁继续使用；本轮没有重新实现 DIALS 的 C++ 反射枚举器。

### 三个值得优先审阅的机制

**物理导数直接参与求解。** 对 Ewald 约束隐式求导得到扫描角变化，
再用商法则得到面板坐标变化。体素中心导数通过概率质量的边界通量计算，
消去固定位置差分步长和六次体积分。几何相切和无法分辨的导数仍明确失败。

**压缩整个统计目标，而不是只压缩一个答案。** 增广 QR 对任意试探强度保留
WLS 目标，包含参数无关残差。严格 Poisson 充分统计量也保留完整参数相关似然。
对于独立计数组，固定总强度后的 KKT 方程为
T_g = N_g / (e_g + lambda)，多维 nuisance 优化变成一个单调标量根；
零计数组的活跃边界解析处理，不加伪计数。

**缩放先消去线性强度。** 每个尺度试探点都先求有符号强度的条件最优值。
完整剖面残差导数保留残差相关的伴随项，在有噪声时不能省略。
固定支撑图只拆分线性子问题，全部尺度仍共享全局信息，
最终协方差仍按完整联合模型计算。

这些是 Ariadne 中实际新增的解算机制。隐函数求导、QR、充分统计量及变量投影
本身是已有理论，本轮不把它们宣称为首次发现的数学定理。

## 两个已经修复的准确性错误

### 大常数吞掉条件似然比

一个条件 Gaussian 幅度约为 3.6、方差为 1 的反例，
参数无关背景 chi-square 为 1.6e17。
独立解析 95% 区间为 [1.64003598, 5.55996395]。
第一轮返回 [0, 7.47298331] 且 status=ok，上端 LR 实际为 0。
本轮先形成目标增量再求和，恢复解析区间，并验证两端 LR 达到阈值。

### 仅改变 ADU 单位就把上界变成零

同一个 PG 物理模型的区间上界为 5.78669768137 ADU。
把数据、增益与标准差同时乘 1e-12，
第一轮固定绝对求根容差使上界变成 0，仍报告成功。
本轮给出 5.78669768137e-12，换回原单位后保持一致。
另覆盖 1e8 单位变换。

对应完整证据：
[likelihood_round2_measurements.json](reports/likelihood_round2_measurements.json)。
这两项是可复现的数值错误修复，不能直接替代真实实验的准确度比较。

## 性能与内存证据

单 BLAS 线程、同一用例、相同目标及收敛容差。
内存为输入已分配后的 Python/NumPy tracemalloc 额外峰值，
**不是进程总 RSS**。每个 JSON 保留软件环境、重复计时和数值误差。

| 合成用例 | 第一轮 | 第二轮 | 结果 |
|---|---:|---:|---|
| 原生双晶体 EM，70 观测/6240 预测 | 1.74215 s | 0.35487 s | 4.91 倍；两者收敛，预测调用 118→6。 |
| 几何兼容度 400×128 | 1.31137 s | 0.022174 s | 59.14 倍；相同候选与密度。 |
| 完整混合 200 万对，额外峰值 | 146.02 MB | 21.86 MB | 减少约 85%；该例计时有显著调度抖动。 |
| WLS 12 万像素/16 分量 | 0.207239 s | 0.055391 s | 3.74 倍；额外峰值 87.49→2.41 MB。 |
| 4096 像素/4 独立 Poisson 完整拟合及区间 | 0.209666 s | 0.037043 s | 5.66 倍；参数相关行 4096→4。 |
| 31 个 PG 信号像素＋32768 严格零轮廓像素 | 0.590881 s | 0.211913 s | 2.79 倍；不能外推至所有尾部非零的轮廓。 |
| 1000:1 一般 PG 强弱重叠 | 0.316003 s | 0.311180 s | 基本持平，精确模型保留。 |
| 120 观测块/60 强度/2 尺度完整缩放 | 0.202943 s | 0.100799 s | 2.01 倍；两者收敛，全联合协方差差 4.45e-13。 |
| HKLF5 15000 观测，额外峰值 | 167.88 MB | 1.33 MB | 减少约 99.2%；正常尺度输出逐字节相同。 |
| 同一 HKLF5 导出耗时 | 1.18976 s | 1.47175 s | 两遍读取，约慢 24%。 |

轮廓专项比较：

| 用例 | 第一轮 | 第二轮 | 结果 |
|---|---:|---:|---|
| 1064895 体素相关 Gaussian 值 | 2.68427 s | 1.23784 s | 2.17 倍；额外峰值 51.39→14.23 MB。 |
| 143715 体素 Gaussian 值＋三轴位置导数 | 2.41212 s | 0.40440 s | 5.96 倍；独立积分验证导数。 |
| 200 参考、4 局部节点原生轮廓学习 | 0.292760 s | 0.122458 s | 2.39 倍；额外峰值 29.05→3.60 MB。 |

模板最大差 5.20e-18，局部协方差最大差 1.78e-15，参考 provenance 一致。
较小 Gaussian 用例存在计时抖动，完整逐次 wall/CPU 数据均保留。

缩放例中线性分量矩阵总元素从 14400 降至 720，
但包含最终协方差等阶段的完整拟合峰值仅从 1.4109 MB 降至 1.3432 MB，
约下降 4.8%；不能把局部矩阵缩小 20 倍写成总内存降低 20 倍。

逐项记录：

- [geometry_round2_measurements.json](reports/geometry_round2_measurements.json)
- [profiles_round2_measurements.json](reports/profiles_round2_measurements.json)
- [streamed_wls_measurements.json](reports/streamed_wls_measurements.json)
- [likelihood_round2_measurements.json](reports/likelihood_round2_measurements.json)
- [variable_projection_measurements.json](reports/variable_projection_measurements.json)
- [export_round2_measurements.json](reports/export_round2_measurements.json)

## 验证与保留的科学边界

完整第二轮原生验证见
[round2_validation.json](reports/round2_validation.json)。
新增原理测试与所有已有测试一起执行，覆盖实际 DIALS experiments、
反射预测和 shoebox 准备，以及数据/工具/进程契约。

**201 项全部通过；0 失败、0 错误、0 跳过。**
相对第一轮新增 50 项测试。原生环境为 Python 3.12.16、NumPy 2.5.4、
SciPy 1.18.1、DIALS 3.30.1、dxtbx 3.30.0、cctbx-base 2026.9。
DIALS distribution 元数据 3.30 与 conda release 3.30.1 分别记录。
必要的 ruff 未定义名称/导出错误检查也通过。

- 新几何导数对应固定仪器、静态晶体及原生选择的局部反射分支；没有扩大为新的全局仪器校准器。
- 完整候选集合保留；分块减少临时数组，输出责任度本身仍为 O(NM)。
- Gaussian 求积阶数规则不构成所有协方差下的严格误差证书；工作预算不包含输入输出。
- WLS 输入轮廓和完整拟合输出仍驻留；位置干扰分支的首个 QR 仍为密集步骤。
- 有符号 WLS 不截零；计数似然保留物理非负强度约束，两种统计对象的语义没有混用。
- 背景仍按原 WLS 规则估计后冻结，区间没有新增背景、轮廓或校准不确定度的联合传播。
- LR 阈值仍是 chi-square(1) 渐近阈值，低计数和非负边界不保证精确覆盖率。
- 缩放变量投影要求线性强度设计在局部满列秩；否则回退原路径，并保留退化诊断。
- HKLF5 无法承载完整协方差；主 JSONL 继续作为完整记录。共同尺度不增加统计信息。
- 未取得用户真实原始实验帧，本轮不报告真实弱峰召回率或结构解析成功率的改善。

## 安装与升级

完整 ZIP 内是成套源码，可以沿用原安装器建立新运行时：

~~~sh
sh scripts/install_local.sh /path/to/conda-dials-3.30.1 /path/to/new-runtime
~~~

对原始 Git 基线已有工作树，从本交付包运行：

~~~sh
python scripts/apply_numerical_update.py /path/to/Ariadne
python scripts/apply_numerical_update.py /path/to/Ariadne --apply
~~~

对**已应用第一轮交付**的工作树，选择第二轮增量：

~~~sh
python scripts/apply_numerical_update.py /path/to/Ariadne --from-first-delivery
python scripts/apply_numerical_update.py /path/to/Ariadne --from-first-delivery --apply
~~~

默认先检查；--apply 才写入。脚本对不兼容的本地修改失败，不做 reset，
不创建 commit 或推送。若用户已另行提交第一轮，使 HEAD 改变，
可审查分支后加 --allow-different-base；补丁上下文检查仍执行。

- patches/ariadne-first-principles.patch：原始 Git 基线 → 当前累计更新。
- patches/ariadne-round2.patch：第一轮交付 → 当前第二轮更新。
- 两种补丁均包括源码、测试、文档和相应基线归档；编译 PDF 在完整 ZIP 中。
- 已复制安装 python/ 的运行时需重新安装该包，避免新 companion scripts 混用旧 Python 包。

## 新算法的调用

其他内核更新自动接入原路径。新的全局变量投影求解器为显式选项：

~~~sh
python skills/ariadne-reduction/scripts/global_scale.py \
  integrated.jsonl geometry.json scaled_result --optimizer profiled
~~~

第三个位置参数是新输出目录；其中写入 scaling.json、corrections.json
及 corrected/ 下的数据，目录应尚未存在。

Python API 新增 optimizer="profiled" 关键字。
默认 optimizer="alternating" 保留已有行为。新路径也使用原有收敛、
秩、鞍点和不确定度验收。未被消费的诊断矩阵只在需要时构造。

Gaussian 内核新增可选参数：

~~~python
values, dvalues = gaussian_voxels_with_derivatives(
    covariance_xyz, centre_xyz, bounds_xyz,
    max_working_bytes=8 * 1024**2,
    out=profile_array,
    derivative_out=derivative_array,
)
~~~

输出 values 为 ZYX，dvalues 为 (XYZ, ZYX)。
out 与 derivative_out 必须为相应形状的可写 float64 数组且不能互相覆盖。
solve_joint 的 chunk_rows=None 默认自动切换；
chunk_rows=0 强制原密集 WLS，正整数指定 QR 的原始行块长。

## 可复现命令

在项目根目录中，使用已装依赖的 Python：

~~~sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  scripts/run_numerical_validation.py --scope numerical \
  --output reports/local_round2_validation.json
~~~

原生环境需包含 DIALS 3.30.1、dxtbx、cctbx、mcp 和已应用的 DIALS 源码覆盖层：

~~~sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  scripts/run_numerical_validation.py --scope native \
  --output reports/local_native_validation.json
~~~

六个基准均自带随机种子和冻结旧模块。建议在原生环境下执行：

~~~sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/geometry_round2.py --repeats 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/profiles_round2.py --repeats 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/streamed_wls.py --repeats 5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/likelihood_round2.py --repeats 5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/variable_projection.py --repeats 7
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/export_round2.py --repeats 3
~~~

geometry_round2.py --skip-native 可只测纯数值部分；
轮廓基准包含实际原生 shoebox 学习，完整运行需要原生环境。
所有基准可用 --baseline-root 指向用户留存的第一轮源码，
也可默认读取 benchmarks/baseline-first-principles.zip。
每个基准支持 --output，便于保留随附证据而另存本机结果。

LaTeX 使用标准 CTeX/Fandol 和 XeLaTeX，编译两遍：

~~~sh
xelatex -interaction=nonstopmode -halt-on-error reports/ariadne_round2.tex
xelatex -interaction=nonstopmode -halt-on-error reports/ariadne_round2.tex
~~~

报告自包含公式、表格及参考文献，不依赖外部图像或 BibTeX 文件。
