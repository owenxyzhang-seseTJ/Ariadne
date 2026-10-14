# 软件后端与环境契约

## 优先级

按当前用户要求：**开源优先，目标和观测正确性优先于软件偏好**。默认 DIALS/cctbx/dxtbx，加 FabIO（图像格式辅助）、Gemmi（结构/反射文件）、NumPy/SciPy/matplotlib。保留本机已合法安装的 SHELXT/SHELXL/PLATON 作为下游解析工具；它们不能被描述成开源软件。Bruker/Rigaku 商业后端仅作为开源路线无法完成特定数据处理时的可选兜底，不是默认必装依赖。

格式读取、指标化、重叠积分、吸收校正、格式导出是分开的能力。列出 installed / configured / demonstrated 状态，不能因找到程序就宣称支持所有仪器。当前交付不运行测试数据，demonstrated 保留 unknown。

## DIALS 开源主路线

在独立环境中配置 DIALS/cctbx/dxtbx、FabIO、Gemmi、NumPy、SciPy、matplotlib。优先官方 DIALS 发行或适用架构的维护包，不用未确认的软件名拼出“已安装”命令。Python 库不代替完整 DIALS 命令套件。

运行真实任务前记录安装版本，并读取该版本命令参数，尤其 `dials.import/index/refine/integrate/symmetry/scale/export`。检查多轴测角仪和探测器 reader 的实际支持。FabIO 解码像素不代表已获得可信实验几何。

参考 DIALS 官方的小分子路线（当前版本参数仍需核实）：

```bash
dials.import /absolute/frame1.cbf /absolute/frame2.cbf output.experiments=imported.expt
dials.find_spots imported.expt output.reflections=strong.refl
dials.index imported.expt strong.refl joint_indexing=True
dials.refine_bravais_settings indexed.expt indexed.refl
dials.reindex indexed.refl change_of_basis_op='a+b,-a+b,c'
dials.refine selected-setting.expt reindexed.refl
dials.integrate refined.expt refined.refl
dials.symmetry integrated.expt integrated.refl
dials.scale symmetrized.expt symmetrized.refl
dials.two_theta_refine scaled.expt scaled.refl p4p=scaled.p4p
dials.export scaled.expt scaled.refl format=shelx shelx.ins=compound.ins shelx.hklout=compound.hkl composition=CHNOZn
```

这是程序关系示意，不是固定一键脚本。每个阶段需选取真实的上一阶段产物。换基操作来自 `bravais_summary.json` 的 cb_op，不是 `.expt` 文件名。composition 优先提供已知元素，不用强行猜 formula/Z。二倍角精修得到的晶胞/esd 应按当前版本导出能力传入匹配 INS，保留与强度索引的基准关系。

对于配位化合物等不能假定 Sohncke 群；`dials.symmetry` 的缩放群/建议不是结构空间群的最终裁决。保留候选和消光证据，下游 SHELXT 比较中心/非中心模型。SADABS 格式未合并导出是可选路线，需要各 run 独立编号及方向余弦；不要对已完成吸收校正的结果重复校正。

### 多晶格和重叠能力

`max_lattices` 只控制索引搜索，reflection experiment id 不等于 crystal domain。将每个晶畴的 U/B/A、约化晶胞和基变换独立记录。

本机 DIALS 3.30.1 已有 detector-space deconvolution 配置及多轮廓拟合代码；不应将其描述为完全不支持重叠。本版可使用 `assets/native-overlap.phil`，也实现了独立同曝光原始像素联合积分器，见 overlap-integration reference。原生集成流程仍不能一律宣称等价 TWINABS：完整观测组、跨分量协方差和全局 composite 缩放须分别检查。不得在标准 scaled.refl 上用近邻搜索拼成 HKLF 5。

DAFi 的原作者仓库实际仅提供 README，发布形式是 Windows 可执行程序，许可限个人/研究/非商业使用，不能称为 OSI 开源或假定有 macOS 源码可编译。当前开源主环境不依赖它。仅有真实多晶集合体需求且平台/许可适合时接入其官方二进制，不安装同名的数据同化 DAFI 项目。DAFi 会覆写 tabbin，必须只对工作副本操作。

## 可选 Bruker 路线

APEX/CELL_NOW 等确定各 orientation；SAINT 多矩阵积分；单晶用 SADABS，非面网孪晶根据软件支持用 TWINABS。确切程序名、输入项目格式、启动命令及日志输出由本地版本官方文档/GUI 实际导出决定。Bruker `.sfrm` 需保留背景/掩膜、实验头和连续扫描；这些校准文件不能当成衍射图。

## 可选 Rigaku 路线

先识别仪器和采集套件。CrysAlisPro 支持自动/半自动处理和脚本接口，但实际 CLI 语法仍须从所装版本确定。保留 `.par` 等几何和 orientation 文件、native raw 帧、探测器校正和日志。不能假定 Rigaku 全部数据都是 CrysAlisPro，或把任意 `.h5` 当单一入口。

## 适配器接口

适配配置保存工作区内，不把本机路径写进可分发 skill。最小配置：

```json
{
  "backend": "dials",
  "executable_root": "/actual/installed/bin",
  "software_version": "actual version",
  "capabilities": {
    "raw_frame_reading": "unverified",
    "multi_lattice_indexing": "documented",
    "overlap_group_integration": "unverified",
    "absorption_correction": "documented",
    "hklf5_export": "unverified"
  },
  "commands": {},
  "parameter_reference": "installed help or official documentation"
}
```

commands 采用 workflow 的 argv 契约。Windows 程序在真实 Windows 主机运行，不能把 WSL 当作 Windows 应用已安装。远程主机/许可证配置由用户指定；未经授权不改采集仪器控制设置。本机安装记录在独立的 `scxrd-environment/runtime.json`；分发包不内置绝对机器路径。
