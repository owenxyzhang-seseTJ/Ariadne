# 复杂孪晶与弱衍射：数学核更新与复现

本文保留第一轮的方法与验证记录。第二轮进一步优化及最新使用说明见
[ROUND2_OPTIMIZATION.md](ROUND2_OPTIMIZATION.md)；
当前累计补丁包含两轮修改，第二轮另附从第一轮升级的增量补丁。

本更新基于 Ariadne 提交 `797ac22ffa9b81da2dd977e013bea8f114b49eb2`，
DIALS 源码接口目标保持 **3.30.1**。本次修改已写入完整源码，附可应用补丁、
原理测试、独立参考结果及中文 LaTeX 推导。

## 核心改变

**局部无法拆开的强度，不等于局部观测没有信息。** 原接口在局部退化时可能只保留
一个总强度。本更新把固定 Gaussian 像素模型中所有数值可观测的强度组合传给
全局模型，允许不同重叠几何共同约束晶畴强度。

对于背景投影后的白化模型 `t = D I + noise`，令
`D S^-1 = U Σ Vᵀ`，输出 `z = Uᵀ t`、`H = Σ Vᵀ S` 及单位模式协方差。
在保留的数值子空间内，像素目标与模式目标仅差一个与强度无关的常数。
这个常数及原像素自由度一起传递，防止压缩后误报更好的拟合优度。

例如，两个局部区域分别提供 `(I1+I2, I3)` 和 `(I1, I2+I3)`。
各区域的局部秩只有 2，但联合秩为 3；若各自只传总量，联合秩仍然只有 1。
新的 `information_modes` 自动接入 `global_scale.py`。已有单分量、总量、
未解决状态及导出约定保留，局部不可辨识的分量不会被伪装成独立 HKLF 测量。

其他已落地的变化：

| 领域 | 修改与用途 |
|---|---|
| 可辨识性驱动的观测选择 | 分析一般设计矩阵零空间，优先选择增加秩的候选块；单位双峰和可解释为无符号 Laplacian。 |
| 弱强度统计 | 保留有符号测量；报告条件信噪比及分量可测信息；精确计数似然仍使用原有物理非负约束。 |
| Poisson–Gaussian | 同一精确卷积的批量有界求和、稳定 Poisson log mass、同似然观测曲率；曲率未分辨时保留剖面区间并不给有限方差。 |
| 晶格评分 | 整数幺模约化；正交度量的分离 theta 和；最近节点的独立几何排除界；整数仿射 HKL 秩。 |
| 全局缩放 | 固定协方差预白化、列均衡 SVD、直接线性响应、稳定指数目标增量。 |
| 可信验收 | 方向局部的算术抵消审计；精确 Hessian 的负曲率驻点检查。 |
| 原始数据归属 | 支持并集代替矩形外壳全填充；审计实际文件、读取帧号及面板，阻止重复使用同一像素。 |

## 实测结果

完整原生验证：**151 项通过，0 失败、0 错误、0 跳过**。
环境是 Python 3.12.16、DIALS 3.30.1、dxtbx 3.30.0、cctbx-base 2026.9，
测试涵盖原生预测和精修、积分、缩放、导出及进程/工具约定。
完整记录为 [first_principles_validation.json](reports/first_principles_validation.json)。
DIALS 的 Python distribution 元数据为 `3.30`，对应 conda 安装记录为 `3.30.1`；
验证器明确记录这两种版本证据。

以下为同一场景、同一软件环境、单 BLAS 线程下的核级中位计时。
基准环境与原生接口测试环境分别记录在各 JSON 内。

| 合成场景 | 基线 | 更新 | 同任务加速 |
|---|---:|---:|---:|
| 4096 像素精确 PG 卷积，259327 求和项 | 4.78892 s | 0.07698 s | 62.21 倍 |
| 5000 点正交宽晶格评分 | 0.03038 s | 0.01104 s | 2.75 倍 |
| 5000 点耦合宽晶格评分 | 0.01955 s | 0.01128 s | 1.73 倍 |
| 72 块相关缩放，精确合成数据 | 0.09133 s | 0.06102 s | 1.50 倍 |

计时包含逐次结果，不表示整套原始衍射处理的吞吐率。噪声缩放例中基线未达到
相同收敛标准，因此不把其时间比写成等价任务加速。

三晶畴实验的完整幅度为 `(1200, 12, 0.8)`。每种固定几何先平均 40 次
独立 Poisson 重复，再把两幅各 120 像素的平均图压缩为 4 个模式。
固定已知方差下，模式解与原像素联合解的最大强度差为 `2.27e-13`，
协方差差为 `3.55e-15`。500 次 Monte Carlo 的最弱幅度均值为 `0.79295`，
其平均偏差的 Monte Carlo 标准误为 `0.03842`；有 17.4% 的弱测量为负。
将同一批弱测量截零会产生 `+0.07622` 的平均偏差。
这是已知几何、轮廓和噪声模型下的机制验证，不是未知晶畴检出率。

## 使用完整源码包

解压后可按原项目方法安装：

```sh
sh scripts/install_local.sh /path/to/conda-dials-3.30.1 /path/to/new-runtime
```

第二个参数应是全新的运行时目录。安装器使用匹配的原生 DIALS 二进制，
应用原有 DIALS 源码补丁，并安装更新的 `python/` 包。
安装动作本身不运行科学测试。

已有运行时若复制安装过 Python 包，更新 Git 工作树后必须同时重新安装
`python/` 或建立新运行时。`python/src/ariadne` 与 companion scripts
应成套更新，避免载入旧包与新脚本的混合版本。

## 对已有 Git 工作副本应用补丁

从本交付包运行应用器，目标参数指向待更新的 Ariadne Git 根目录：

```sh
python scripts/apply_numerical_update.py /path/to/existing/Ariadne
python scripts/apply_numerical_update.py /path/to/existing/Ariadne --apply
```

默认仅检查，显示将修改的文件。`--apply` 在兼容性检查后调用 `git apply`；
不提交或推送，不使用 reset，不忽略不相容的本地修改。
默认要求目标 HEAD 为上述基线；若你的分支已经增加其他提交，可在审阅后加
`--allow-different-base`，仍必须通过完整上下文检查。
再次运行时若完整补丁已经存在，会明确报告而不重复应用。

补丁位于 `patches/ariadne-first-principles.patch`。
它包含源码、测试、文档和用于对照的基线归档。
完整 ZIP 另外包含已编译的报告 PDF。

## 复现测试与实验

纯数学核需要 NumPy/SciPy，原生部分需要匹配的 DIALS 运行时。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/run_numerical_validation.py --scope numerical

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ariadne-env python scripts/run_numerical_validation.py --scope native

OPENBLAS_NUM_THREADS=1 python benchmarks/twin_information.py \
  --output reports/twin_information_reproduced.json
OPENBLAS_NUM_THREADS=1 python benchmarks/lattice_principles.py \
  --output reports/lattice_reproduced.json
OPENBLAS_NUM_THREADS=1 python benchmarks/pixel_principles.py \
  --output reports/pixel_reproduced.json
OPENBLAS_NUM_THREADS=1 python benchmarks/scaling_principles.py \
  --output reports/scaling_reproduced.json
```

数值测试不下载图像，原生测试用合成 DIALS 实验。
完整测试中的进程生命周期约定需要正常的本地进程可见性。
`benchmarks/baseline-797ac22.zip` 提供固定基线，不要求 ZIP 解压副本具有 Git 历史。

候选观测选择示例：

```sh
python skills/ariadne-reduction/scripts/overlap_observability.py \
  examples/overlap_design.json --budget 1 --output selection.json
```

它从给定设计中选择能补充缺失方向的观测。实际应用需输入预测几何给出的
真实设计矩阵及噪声协方差。当前块和候选块之间假定独立，块内允许相关误差。
脚本是观测设计诊断，不操作采集硬件。

## 科学适用范围

1. 信息模式对固定轮廓、固定权重的 Gaussian 二次目标，在保留的数值子空间内
   保持参数相关信息。默认从观测值估计的像素方差属于数据依赖 plug-in 权重；
   条件代数恒等式不等于精确 Poisson 似然的充分统计量。IRLS 模式不会被当作
   精确固定协方差模式，非 Gaussian 估计器不走这条等价压缩路径。
2. 不同晶畴、独立晶体和不同相的强度不能自动视为相同。全局恢复要求用户提供
   正确的等效强度身份、实际几何及允许共享的物理模型。
3. 精确 PG 分支允许校正后负 ADU，仍保持非负光子期望。
   当前背景先由 WLS 估计后固定；PG 区间不包含背景估计不确定度。
   剖面似然采用渐近阈值，不承诺低计数边界的精确覆盖率。
4. 全局缩放协方差与数据响应采用 Gauss–Newton 局部线性化。
   精确 Hessian 用于拒绝负曲率驻点，不将该协方差升级成完整非线性误差传播。
   缩放优化不保证全局最优，也不自动逃离已诊断的鞍点。
5. LLL、SVD、Poisson 求和与无符号 Laplacian 是已有数学工具。
   本次方法贡献是 Ariadne 中新的信息保留、联合恢复和观测选择机制，
   不能据此宣称首次发明这些数学原理。
6. 未用用户实际 Bruker/Rigaku 原始弱衍射和复杂孪晶数据完成端到端精度验证。
   原有 spot proposal 保留，当前结果不建立真实弱点检出率提高的结论。

## 数学说明

[ariadne_first_principles.tex](reports/ariadne_first_principles.tex) 是完整、
可编辑的中文数学报告，包含模型、推导、图机制、反例、实测和参考文献。
它使用标准 `ctexart`、Fandol 字体和内嵌 TikZ 图；安装 TeX Live 中文支持后：

```sh
xelatex -interaction=nonstopmode reports/ariadne_first_principles.tex
xelatex -interaction=nonstopmode reports/ariadne_first_principles.tex
```

全部数据表的机器可读依据在 `reports/*principles*json` 与
`reports/twin_information_experiment.json`。
