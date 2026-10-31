# 技术报告编译

源码包已提供 `docs/report.pdf`。可编辑源文件为 `docs/report.tex`、`docs/chapters/` 和 `docs/references.tex`；报告插图位于 `docs/figures/`，以矢量 PDF 提供。直接阅读 PDF 不需要安装 LaTeX 或 Python。

## 编译环境

报告使用 XeLaTeX，建议安装 TeX Live 或具有相同组件的发行版。需要 `geometry`、`fontspec`、`amsmath`、`amssymb`、`mathtools`、`bm`、`graphicx`、`booktabs`、`longtable`、`array`、`tabularx`、`multirow`、`xcolor`、`enumitem`、`fancyhdr`、`titlesec`、`tikz`、`hyperref` 和 `listings`。

中文字体为 Noto Serif CJK SC，西文辅助字体为 Latin Modern Roman 和 Latin Modern Mono。后两者通常由 TeX Live 提供。数学公式和文字保留为原生 LaTeX，不需要栅格页面或 Word 模板。

## 字体准备

若系统已经安装 Noto Serif CJK SC，可以直接编译。否则在**项目根目录**执行：

```bash
python tools/fetch_report_fonts.py
```

该辅助脚本使用 Python 标准库，从官方 Noto 仓库下载 Regular、Bold 两个字体和 SIL Open Font License，保存到 `docs/fonts/`，并记录下载来源及哈希。此步骤需要网络。字体本体不包含在源码包中；LaTeX 会在找不到系统字体时使用该本地目录。

如果使用已有的其他中文字体，可修改 `report.tex` 顶部的 `fontspec` 字体选择。替换字体可能改变分页和行宽，修改后应查看生成的 PDF。

## 生成 PDF

从项目根目录执行：

```bash
cd docs
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

结果为 `docs/report.pdf`。第二次编译用于更新目录和交叉引用。参考文献由 `references.tex` 直接提供，不需要额外运行 BibTeX。报告使用随包插图，因此仅编译报告不要求安装 Python 数值依赖。

## 重新生成计算与图形

如需重新运行数值示例，先按 [README 安装说明](../README.md#安装)安装 Python 包，然后在**项目根目录**执行：

```bash
python examples/generate_examples.py
ariadne-xtal run examples/refine_job.json --root .
python validation/run_real_fen4.py
```

第一个命令生成模拟输入，第二个命令在新的 `runs/` 子目录中输出精修模型、交互 HTML、密度图和静态图。实测回归脚本更新 `validation/real_fen4/` 下的模型与数值记录。两者不会自动替换报告中的插图或数值表；如改变算法或数据，应依据新结果明确更新 `docs/figures/` 及相应 LaTeX 章节。

需要另外生成静态图或离线仪表板时，使用 [API 文档中的绘图接口](API.md#6-密度图与绘图)。交付中的报告及验证数值对应随包保留的计算快照。
