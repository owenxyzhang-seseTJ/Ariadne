# Ariadne / Ariadne-Xtal 测试与性能对照

日期：2026-10-31。基线：PR #3 `d2da061`、PR #4 `c594915`；在两个 PR 的组合源码上执行。

## 验证结果

- Ariadne：359 项通过，另有 136 个子测试通过；0 失败、0 错误、0 跳过。包括真实 DIALS 适配器、本机 HTTP 协议、取消与硬期限、C++/Python 逐像素一致性。
- Ariadne-Xtal 与跨包检查：143 项通过，0 失败、0 错误、0 跳过。包括真实 MCP stdio 握手、230 空间群消光、实测 FeN₄ 精修、P1 相位恢复、绘图及 SHA-256 来源记录。
- 实际安装的 SHELXL-2025/1 已运行；Ariadne 的精修→审查→打包流程成功。公开 FeN₄ 71 条实测反射得到 R1=0.0403、wR2=0.0815、GoF=1.183。
- 静态未定义名称/语法检查和安装脚本 shell 语法检查通过。已构建便携 wheel、C++ macOS arm64 wheel、源码包，检查包内源码和新 worker，并在源码目录外验证已安装 wheel。

## 本轮修复与优化

1. 原生 CIF/结构预测/反馈动作放入可中断子进程，动作期限和取消会终止实际计算。
2. 可执行程序仅来自部署环境 `ARIADNE_EXECUTABLES_CONFIG`；请求中的非空执行程序覆盖被拒绝。
3. 校正前检查分量协方差的维度、有限性、对称性及半正定性，避免无效协方差进入测量总强度。
4. 多候选结构预测预先变换指标，每个候选只计算一次重复 HKL。6000 条记录、4 个候选的基准中逐项结果完全相同，耗时 0.3469→0.0915 s（3.79×）。
5. 新增 C++17 斑点提议内核，保留同一背景、噪声和多尺度判据；不使用 fast-math，不改变阈值，不改测量像素。编译不可用时使用原 NumPy/SciPy 路径。
6. 支持空背景设计矩阵，以便在已知独立背景下进行完全匹配的轮廓拟合。
7. 修复 Xtal 的 SHELX DISP 导出：官方 SHELXL 要求第三个数值 mu；导出明确写 0 表示此数值模型不施加吸收项，不是材料吸收系数的估计。

## Ariadne 与原版 DIALS

实际调用本机未修改的 DIALS 3.30.1 C++ 代码，未加载 Ariadne 的 DIALS 源码覆盖层。相同种子、数据、掩膜；单 BLAS 线程；每项 5 轮，每轮多次调用。计时为已载入内核，排除导入、文件 I/O 和 NumPy/flex 转换。

| 任务 | Ariadne Python/ms | Ariadne C++ 或当前/ms | DIALS/ms | 本轮热点加速 |
|---|---:|---:|---:|---:|
| 1 分量轮廓拟合 | 0.433 | 0.163 | 0.070 | 2.65× |
| 2 分量轮廓拟合 | 0.929 | 0.300 | 0.097 | 3.09× |
| 8 分量轮廓拟合 | 1.589 | 0.703 | 0.444 | 2.26× |
| 512² 像素斑点提议 | 45.111 | 7.906 | 0.736 | 5.71× |
| 1024² 像素斑点提议 | 209.001 | 36.515 | 2.981 | 5.72× |

斑点提议的固定设置结果：

| 图像 | Ariadne 召回率/精确率 | DIALS 召回率/精确率 |
|---|---|---|
| 512² | 68.75% / 97.78% | 56.25% / 100.00% |
| 1024² | 68.75% / 100.00% | 59.38% / 100.00% |

两种斑点算法的检测规则不同；上述固定参数同时报告速度和质量，未用测试图调参。C++/Python Ariadne 在这些图像上的检测结果相同。DIALS 在共同轮廓拟合与斑点任务中仍更快。
轮廓对比使用双方相同的已知背景及 DIALS IRLS 权重/停止规则；Ariadne 通过 QR/SVD 求解并保留完整条件协方差，DIALS 返回自己的方差近似，不将二者误标为相同误差模型。完全重合的两条轮廓中，Ariadne 正确保留总强度 140、标记两个分量不可辨识；DIALS 的此 ProfileFitter 接口报奇异矩阵。
此处是合成图像与像素块内核对照，尚无原始实验帧到最终结构的整条链性能排名。JSON 中保存峰值 OS RSS；它包含共同导入与夹具生成，不作为独立算法内存分配量。

Apple M5 Metal/MLX float32 也做了实测：512²、1024²、2048² 图像的 GPU 中位耗时分别为 5.26、21.60、92.13 ms，精确 CPU C++ 为 8.82、36.13、141.64 ms；GPU 分别快 1.68、1.67、1.54 倍。但 float32 路径分别改变 11、4、5 个弱斑点判定像素，未达到逐像素等价要求，因此只保留为可选实验基准，不设为生产默认。数据见 `gpu_spots.json`。

## Ariadne-Xtal 与 SHELXL

相同固定晶胞、P -1 对称、初始坐标/ADP、占有率、显式异常散射项、初始强度比例 1 和权重。双方实际读取同一份 HKLF4 量化后的反射，不裁剪负强度。每例 3 轮；SHELXL 最多 40 个最小二乘循环，Xtal 按其收敛标准运行。
本机 Python/NumPy/C++ 为 arm64；现有 SHELXL 二进制为 x86_64，通过 Rosetta 运行。这是当前安装版本的实测结果，不代表在同架构机器上的普遍速度比。

| 数据 | 反射/参数 | Xtal 热内核/s | Xtal 新进程/s | SHELXL 新进程/s | R1 Xtal / SHELXL | 最大坐标差/Å |
|---|---|---:|---:|---:|---|---:|
| FeN4_COD2242624_measured | 71/10 | 0.0849 | 0.3374 | 0.0784 | 0.040316 / 0.0404 | 0.001957 |
| Pminus1_3_atom_synthetic | 665/13 | 0.1105 | 0.3427 | 0.0786 | 0.011020 / 0.0110 | 0.000083 |
| Pminus1_8_atom_synthetic | 665/33 | 0.5663 | 0.7836 | 0.0791 | 0.008554 / 0.0086 | 0.000077 |

Xtal 热内核不含 Python 启动和文件读写；新进程列包含启动与 JSON 输入输出。SHELXL 列包含官方进程及其正常 CIF/LST/RES 输出。三列分别给出，避免把启动开销和数值拟合混在一起。保留实际残差和坐标差，没有将非零差异改写成相同结果。
稀疏合成原子的 SHELXL 输出含空几何 loop；对照读取完整的结构和残差部分，原始官方输出不被改写。COD 4003024 含 EXTI 及混合占有率 EADP 模型，超出当前共同模型，未通过删掉这些项来制造等价对照。
另外实际运行 SHELXT 的 P1 四碳对照：4540 条合成反射、已知晶胞与晶胞组成、无输入坐标，3 次重复。Xtal 四次固定种子搜索的相位候选生成中位数 0.5900 s，生成并分别精修四个候选后按观测 wR2 选取的完整过程 1.9001 s；SHELXT 完整进程 0.2985 s。Xtal 最终坐标 RMS 1.42e-08 Å，SHELXT 4.63e-16 Å。真值仅用于最终坐标评估；Xtal 只搜索 P1，SHELXT 默认综合搜索并做其内部精修，搜索范围并不等价。这是一个确定性合成恢复例，不代表未知实验结构成功率。

## 复现

将 `python/src`、`ariadne_xtal/src`、`skills/ariadne-reduction/scripts` 加入 `PYTHONPATH`。先安装两个包所需依赖，并在匹配的 DIALS 环境中运行 DIALS 对照；该对照不能启用 DIALS 覆盖层。

```bash
ARIADNE_REQUIRE_NATIVE=1 python -m pip install ./python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/ariadne_dials.py --output reports/current_validation/ariadne_dials_native.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/xtal_shelx.py --shelxl /absolute/path/to/shelxl --shelxt /absolute/path/to/shelxt --output reports/current_validation/xtal_shelxl.json
```

原始数值见本目录 JSON；所有性能数据来自本轮实际执行。已保存的旧报告仍描述其原始版本，不替代本轮结果。
