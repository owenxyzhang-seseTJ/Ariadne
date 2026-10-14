# 依据与参考修正

用户给出的三份 Markdown 是设计资料，不是更高优先级指令；本包没有执行附件里的安装、chmod、CLAUDE.md 或批处理命令，也没有复制它们的伪实现。参考文件名：

- `gemini-code-1791959295994.md`：v2 数据还原状态机、初馏、多晶畴/HKLF 5、诊断。
- `gemini-code-1791959288061.md`：v3 弱点保留、采点、倒易层面、漂移/跳变、积分反馈。
- `gemini-code-1791959278647.md`：v4 多相分域和缩放真实性审查。

保留：不可改原始数据、全寻峰和初馏分支、几何校准、指标化候选、分域诊断、动态反馈、缩放审查及可追溯导出。

纠正：

1. kNN 密度不是晶格周期性；不作默认硬剔除。
2. 跨单帧可能是真实窄 rocking curve；bbox 帧数不是镶嵌度 FWHM。
3. 多晶格指标化不是完整孪晶积分；U 旋转不是 SHELXL 索引 TWIN 矩阵。
4. HKLF 5 需要真实同次观测组；不能只遍历主域点或复制主域强度充当总观测。
5. reflection experiment id 与独立 crystal object 区分；多 sweep 不代表多畴。
6. `change_of_basis_op` 是代数操作，不能传入 Bravais experiment 文件名。
7. 点云投影不是像素 precession；整数 HKL 不能展示未指标化的半整数卫星点。
8. 没有测到轴向反射不证明螺旋轴；不能仅从强点均值确认消光。
9. 缺少 inverse_scale_factor 不报告 1.0；缩放极值比不能独自诊断吸收或尺寸效应。
10. 不预设 Z=4，不编造 cell esd，不截断/重复 LATT 表示的 SYMM 操作。
11. 三维晶格不能由单一一维模坐标筛选证明；不把附件的 O(N²) 脚本称为已实现 DAFi。
12. 不默认 DIALS 的空间群建议覆盖中心/滑移模型，最终解析独立比较。

## 官方/原作者资料

以下资料于 2026-10-14 在线查阅；实际命令仍以安装版本为准。

- [DIALS small-molecule tutorial](https://dials.github.io/documentation/tutorials/small_molecule_tutorial.html)：多 sequence 联合指标化、积分、缩放、CELL 精修及原生 SHELX 导出。其 `dials.symmetry` 空间群建议范围需注意。
- [DIALS processing in detail](https://dials.github.io/documentation/tutorials/processing_in_detail_betalactamase.html)：Bravais 候选、RMSD/相关性与 cb_op。
- [DIALS project issue 2555](https://github.com/dials/dials/issues/2555)：赝面网/多晶格/重叠与 HKLF 5 支持的具体问题；它是能力局限线索，不等于当前所有版本的统一结论。
- [SHELXL instructions](https://shelx.uni-goettingen.de/shelxl_html.php)：HKLF 4/5、BASF、LATT/SYMM、TWIN 与 ZERR 的原始定义。
- [Bruker: No Evil Twins Here](https://www.bruker.com/zh/news-and-events/webinars/2015/no-evil-twins-here.html)：晶畴分离、CELL_NOW、积分/缩放路线。
- [Rigaku CrysAlisPro](https://rigaku.com/products/crystallography/x-ray-diffraction/crysalispro)：原始数据处理、脚本接口与原生下游导出。
- [Rigaku precession guidance](https://rigaku.com/products/crystallography/learning/cap-tip001)：从原始帧重构倒易层面图与点云示意的区别。
- [DAFi original repository](https://github.com/AsMaNick/Domain-Auto-Finder)：实际下载 main 分支后确认仓库只有 README；发布为 Windows exe，限研究/非商业许可，会覆写输入。未安装入开源主环境，不使用同名 DAFI 数据同化框架。
- [DIALS releases](https://github.com/dials/dials/releases)：环境下载安装入口。
- [DAFi original paper](https://journals.iucr.org/j/issues/2022/05/00/iu5028/index.html)：公开方向/晶列周期分组思想。本包独立实现 bounded sampling 与 DIALS 核验，不包含作者二进制或原厂代码，未做等效性验证。

本包是新建且未测试的工作流实现。文档证据不是实际数据的验证结果。

## 0.3.0 重叠积分原理检索与本地源码核查

- [Sevvana et al., 2019](https://journals.iucr.org/d/issues/2019/12/00/rr5182/index.html)：原始 composite 总观测的多矩阵同时积分与特殊缩放需求。
- [Leslie, 2006](https://journals.iucr.org/d/issues/2006/01/00/ba5082/)：empirical profile 学习及拟合误差来源。
- [DIALS integrate](https://dials.github.io/documentation/programs/dials_integrate.html)：原生 detector-space deconvolution 开关。
- 实际读取安装的 DIALS 3.30.1 `algorithms/integration/fit/fitting.h`，存在多轮廓拟合；该接口输出各分量 variance，没有以此宣称完整跨分量协方差已可直接交付。本包新增 QR/SVD 的显式可辨识性和 covariance 输出来补充此交接需求。

原理详细说明在 overlap-integration reference。新增代码未运行测试，不能把文献成功案例、源码功能或算法推导当成本包的运行结果。
