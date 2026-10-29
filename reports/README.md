# DIALS 数值算法优化与技术报告

`dials_algorithm_optimization.pdf` 与同名 `.tex` 是主报告。面向晶体学专业人员以及懂 AI、尚不熟悉晶体学的读者，涵盖衍射几何、指标化、计数噪声、积分、多晶格 EM、全局缩放、对称合并和本轮算法设计。

本轮实现包括 DIALS 原生候选选择与批量排序、周期 Gaussian 证据的实空间/Fourier 自适应表示、相关 Gaussian 条件解析积分、位置 nuisance QR 复用、Gaussian 模板分派和缩放回调复用。源码和安装补丁保存在本地分支 `dials-numerical-optimization-20261029`。

`optimization_measurements.json` 记录五次测量、中位数、峰值追踪分配及数值误差。倍率属于相应计算核心，不代表整套原始图像约化速度。宽相关峰采用更高精度积分，记录速度比为 0.86；其代表体素绝对误差从 2.91e-4 降到 1.64e-15。带噪缩放压力样例在两版中都达到原有迭代预算，已知真值样例在两版中均收敛；数据文件保留两组状态。

`validation.json` 与 `validation.log` 记录最终科学回归：75 项通过、无失败、无跳过。安装补丁应用后与三个 DIALS overlay 文件逐字节一致。

复现性能测量，在已安装 Ariadne 环境中运行：

```bash
ariadne-env python benchmarks/optimization.py --repeats 5
```

基准按开发源码路径或安装后的 DIALS 模块加载当前实现，旧版固定在 `benchmarks/baseline-2b9bd04.zip`。可用 `--only` 重测指定核心并保留其余结果。

编译报告，在插件根目录运行：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=reports reports/dials_algorithm_optimization.tex
```

本轮产物未推送、未发布，也未覆盖已部署运行时。
