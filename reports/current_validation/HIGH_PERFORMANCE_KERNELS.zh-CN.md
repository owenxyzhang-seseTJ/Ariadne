# 高性能路径重写清单与实测

日期：2026年11月02日

本文回答“所有需要高性能的任务能否写成更高效的内核”。结论按路径分成三类：已重写并实测、保持现有原生实现、以及保留 Python 但通过算法改动提速。所有数字来自本机同进程交替测量，脚本、原始样本和误差保存在 `reports/current_validation/`。这些是子项或合成数据的测量，不等于整条还原链快于 DIALS。

## 1. 已重写并实测

| 路径 | 原实现 | 现实现 | 实测加速 | 数值一致性 |
| --- | --- | --- | --- | --- |
| 单峰检测 matched filter | Python 分步 NumPy | `_native.cpp` 四工作单元融合内核 | 4 线程 36.11 → 24.96 ms（1.45×） | 逐像素 diff = 0，54+4 项测试通过 |
| Poisson–Gaussian 精确卷积 | NumPy 分批 logsumexp | `_likelihood_kernel.h` 逐像素融合求和 | 3.88–5.64×（4096 像素 14.76 → 3.81 ms） | 对数密度误差 ≤ 3.6e-15 |
| 完整 PG 拟合与全部区间 | 同上 + SciPy 求解 | 同上 | 2.75–2.92×（33.2 → 11.6 ms 量级） | 强度 ≤ 1.2e-12，区间 ≤ 1.3e-12，状态全 ok |
| 质心 EM 完整候选责任度 | NumPy einsum + logsumexp | `_centroid_kernel.h` 三维白化残差 | 8.7–12.4× | 绝对误差 ≤ 3.6e-15 |
| 复合缩放设计装配 | 逐贡献列 Python 循环 | `_scaling_kernel.h` 有序累加 | 7.1–29.3×（装配子项），完整拟合 1.32–1.34× | 逐元素完全相同 |
| 重叠框连通分组 | Python active list + 并查集 | `_grouping_kernel.h` int64 精确内核 | 7.8–70.4× | 分组及组内顺序完全相同 |
| 周期行滑窗 | Python 双指针 | 同上，仅编译已排序窗口 | 完整 `common_gap` 7.8×、`periodic_subset` 4.4× | 端点与候选索引完全相同 |
| 缩放侧表准备 | 逐贡献者重复对称化 + 每扫描重扫 | 等效 HKL 缓存 + 单遍范围归约 | 1.42×（单扫描）、1.39×（16 扫描） | 完整输出逐项相同 |

证据：`numerical_kernels.json`、`scaling_kernel.json`、`grouping_kernels.json`、`global_prepare_kernel.json`、`matched_rewrite_timing_final.json`。

## 2. 加速没有改变科学模型

- 不使用 float32、fast-math、降低采样率、减少迭代次数或修改检测阈值。
- Poisson–Gaussian 仍保留每像素自适应窗口、两侧几何尾界和计算预算；超预算抛 `LikelihoodBudgetError`，不返回近似值。
- 质心 EM 保留全部合法候选、完整 3×3 协方差、离群均匀分量；没有近邻裁剪或后验概率阈值。
- 缩放求解保留逐列平衡的 QR/SVD，不改成正规方程，也不加 ridge；秩亏仍只报告可辨识子空间。
- 分组内核使用 int64 精确比较，不经过 double，链式相交和边界接触语义与原实现一致。
- 本轮修正了一处真实缺陷：原生卷积在极端 ADU 单位（gain 1e-12 与 1e8）搭配 1e12 均值时，编译器把乘减融合成 FMA，结果与 NumPy 相差约 2e-5。显式阻止融合后，误差回到 1e-15 量级，相关边界测试已加入回归集。

## 3. 保留原实现及原因

| 路径 | 原因 |
| --- | --- |
| QR/SVD、Cholesky、三角求解、BLAS/LAPACK | 已经是 LAPACK 原生实现；重写无法在保持弱方向与秩判定的前提下更快 |
| `scipy.optimize.least_squares` 外层求解 | 求解器本身是原生代码；剩余时间花在必须精确的模型评估和线性代数上 |
| `cKDTree` 邻近查询 | SciPy 原生 C 实现，逐点调用已批量成一次查询 |
| DIALS/dxtbx 预测、`compute_corrections`、Gemmi 结构因子 | 外部原生库；重写只会复制其行为，且会破坏与 DIALS 的可比性 |
| `profile_geometry` 高斯体素积分 | 已使用 `ndtr`/`erf`/`roots_legendre` 与分块，历史优化已达 1.47–5.96×；再次重写会改变积分阶数或尾部稳定性 |
| `crystal.py` Fraction 有理变换 | 精确整数/有理运算，浮点替代会破坏晶格变换的可逆性 |
| `lattice_math` LLL 与大整数最近格点证书 | 依赖任意精度整数与 longdouble 舍入控制，编译成 double 会失去正确性保证 |
| `hklf5_stream` 流式读写 | 瓶颈在 I/O 与逐记录审计；此前优化把内存从 168 MB 降到 1.33 MB，字节一致，时间未改善 |
| `export_hklf5`、`joint_geometry_refine` 等 CLI 包装 | 不含数值循环，只有参数与 DIALS 调用 |
| 小输入分组/窗口 | 实测 16 框 0.99×、8 点 0.99×；转换与调用开销抵消收益，保留 Python 分支 |

## 4. 测试

在仓库根目录、Python 3.12 科学环境运行 `tests` 与 `tests_rebuild`：456 项通过、138 个子测试通过、0 失败 0 跳过。Python 3.13 下分组与数值内核回归 37 项通过。日志见 `high_performance_full_suite.txt` 与 `high_performance_tests_py313.txt`。

注意：从 `python/` 子目录运行全套件会因示例脚本使用相对路径而产生 13 项与内核无关的失败，须在仓库根目录运行。首次构建确认 sdist 包含全部 5 个原生源文件（`high_performance_sdist.txt`）。

## 5. 覆盖边界

- 上述计时均来自合成或固定的中小规模数据，用于比较实现本身，不是真实扫描的端到端时间。
- matched 检测在四工作单元下仍比 DIALS C++ 慢约 2.14×；本轮没有改变这一事实。
- 完全重叠的孪晶分量在数学上不可辨识；内核只报告总强度与可辨识组合信息，不声称能拆分单个强度。
- 多扫描、多域联合积分、吸收/缩放的整链科学验收不在本轮范围内。

## 6. 复现

```bash
cd python && ARIADNE_REQUIRE_NATIVE=1 /Users/xiaoyuzhang/CrystallographyPrograms/scxrd-open/bin/python setup.py build_ext --inplace
```

```bash
cd python && PYTHONPATH=src:/private/tmp/ariadne-testdeps:../skills/ariadne-reduction/scripts /Users/xiaoyuzhang/CrystallographyPrograms/scxrd-open/bin/python -m pytest ../tests_rebuild/test_numerical_kernels.py ../tests_rebuild/test_grouping_kernels.py -q
```

```bash
cd python && PYTHONPATH=src /Users/xiaoyuzhang/CrystallographyPrograms/scxrd-open/bin/python ../benchmarks/numerical_kernels.py --repeats 7 --output ../reports/current_validation/numerical_kernels.json
```

```bash
cd python && PYTHONPATH=src /Users/xiaoyuzhang/CrystallographyPrograms/scxrd-open/bin/python ../benchmarks/scaling_kernel.py --output ../reports/current_validation/scaling_kernel.json
```

```bash
cd python && PYTHONPATH=src /Users/xiaoyuzhang/CrystallographyPrograms/scxrd-open/bin/python ../benchmarks/grouping_kernels.py --repeats 7
```

```bash
cd python && PYTHONPATH=src /Users/xiaoyuzhang/CrystallographyPrograms/scxrd-open/bin/python ../benchmarks/global_prepare_kernel.py
```
