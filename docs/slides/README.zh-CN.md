# 单晶衍射：晶体学基础与计算方法

中文课堂课件，采用黑白教材式线图、宋体正文及本地 KaTeX 公式。内容参照用户提供的《Ariadne Monograph》，补充从结构基元与点阵开始的入门章节。

在线入口：https://owenxyzhang-sesetj.github.io/Ariadne/slides/

## 内容

- 结构基元、点阵、晶胞、原胞与惯用胞。
- 六晶族、七晶系、七点阵系与十四种 Bravais 点阵；七晶系的并列晶胞图。
- 晶向、Miller 指数、倒易基矢与晶面间距。
- Bragg 光程差、分辨率、Laue 条件、Ewald 球和旋转法。
- 结构因子、系统消光、相位恢复与受约束精修。
- Ariadne 数据还原、重叠积分、QR 压缩、结构反馈、agent 与已有证据。

## 使用

离线使用：下载 `Ariadne_Introduction_Offline.zip`，解压后打开 `slides/index.html`。PDF 讲义为 `slides/Ariadne_Crystallography_Lecture.pdf`。包内技术报告及 LaTeX 工程保持相对链接。无需联网、npm 安装或构建。

方向键 / PageUp / PageDown 翻页，Home / End 跳转首尾；N 切换备注，M 打开目录，F 请求全屏。也可使用页面按钮。目录提供打印入口。浏览器可能限制本地文件的全屏功能。

Bragg 图可调 θ 并显示光程差；Ewald 球可调晶体旋转角 φ，播放或复位到衍射位置；重叠轮廓可调中心间距。工具栏可暂停动画，系统减少动态效果偏好同样生效。

图形是基于数学关系新绘的教学示意，不代表真实实验。Bragg 动画仅示意波前传播；任意滑块角度不自动满足衍射条件。Ewald 点阵绕固定倒易原点旋转，球面保持不动。数值页引用仓库既有记录，本次制作没有重跑科学实验。

## 符号与许可

本课沿用专著约定：B 为直接晶胞矩阵，A=B⁻ᵀ 为不含 2π 的倒易基矩阵。56 页报告的部分章节采用另一 A/B 命名，阅读时按对象定义转换。

公式引擎 KaTeX 0.19.1 的静态文件与字体位于 `vendor/katex/`，MIT 许可证原文保留于 `vendor/katex/LICENSE`。所有公式均离线渲染为 HTML + MathML，图形为本地 SVG / Canvas。用户提供的参考专著没有另行上传。

离线包不包含实测反射表、原始帧、数据库或结构候选。完整理论、接口边界与复现口径见相邻 `technical-report/` 目录的报告。
