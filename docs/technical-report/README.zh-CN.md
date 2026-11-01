# Ariadne 数学与算法技术报告

报告日期：2026年11月01日。

主文件为 `Ariadne_Mathematical_Technical_Report.tex`，PDF 为同名文件。正文含 15 章、复现附录、主要符号、目录与参考文献。报告参考用户提供的 47 页 `Ariadne_Xtal_Mathematical_Technical_Report.pdf`，复用仓库保存的相应理论章节与三个已有合成图版，并新增原始像素还原、matched-filter、晶格评分、共享像素积分、QR 压缩、低秩反馈、实现契约与最新证据章节。

## 编译

需要支持 XeLaTeX 的 TeX Live，使用随 TeX Live 提供的 ctex 与 Fandol 字体。无需下载字体，也无需启用 shell escape。

在当前目录运行以下命令两次，以生成目录与交叉引用：

```bash
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error Ariadne_Mathematical_Technical_Report.tex
```

`chapters/` 和 `figures/` 必须保留相对位置。`references.tex` 为参考文献，不依赖 BibTeX。

## 材料与结果范围

源码基线为 `cec63ba186a1843389b1ae161944388304ff1cdd`。数值来自仓库 `reports/current_validation/` 的既有记录；本次报告编写没有重跑科学基准或新增结构验收。历史记录的环境和计时口径分别解释，原报告中“尚未运行官方程序”的旧叙述已按现有真实 SHELXL 对照更新。

报告明确保留未完成事项：当前 matched 检测速度仍低于 DIALS C++，全扫描多域约化与新 raw 路线完整结构解析尚未建立科学接受；Gaussian QR 的加速结论限于所列固定方差合成组。

交付包包含本报告 PDF、LaTeX、参考文献、三个既有图版和本说明，不包含实测反射表、原始帧、运行数据库及结构候选文件。发布目录为仓库的 `docs/technical-report/`；配套交互演示位于 `docs/slides/`。未执行任何 SHA-256 计算或验证。
