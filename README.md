# Ariadne · 0.5.0

面向 Bruker/Rigaku 旋转单晶原始数据的开源优先、Agent 原生工作流：多晶格假设 → 非互斥几何匹配 → 受约束精修 → 全反射预测 → 联合重叠积分 → 真实校正 → INS/HKL → solve-single-crystal。

在 ReciprocalForge 0.4.0 基础上改名并完善。参考用户提供的概率多晶格方法论，复用 DIALS/dxtbx/cctbx 的几何、预测和精修，增加假设/观测的多对多表示；没有将附件研究构想宣称为已验证方法。

| 能力 | 本版范围 |
|---|---|
| 多晶格候选与软解释 | 原 DAFi-inspired/FFT 搜索，加 q 与原生 mm/mm/rad 预测双重匹配；保留非互斥解释 |
| 几何更新 | 原生 DIALS joint refinement，用唯一锚点软加权；固定共同仪器，多源质心不伪装独立观测 |
| 留出比较 | 按扫描块留出，固定行比较 coverage/compatibility/独立支持；保留备选，不伪称 Bayes evidence |
| 局部轮廓与重叠 | 正则化局部 empirical profile、共同像素连通组、QR/SVD、NNLS/Poisson 可选估计器 |
| 不确定度与交接 | 可辨识 total/component、条件 covariance、非负边界标记、partiality 定义、phase-aware HKLF5 与有损导出说明 |
| Agent 接口 | 9 个 MCP tools、固定 scientific adapter 枚举、异步预算、immutable hypotheses、append-only events |

源码入口：`skills/ariadne-reduction/SKILL.md`。详细实现、输入契约和限制见 [概率多晶格接入说明](skills/ariadne-reduction/references/probabilistic-multilattice.md)；底层 patch 与峰形见 [DIALS 源码扩展](skills/ariadne-reduction/references/source-extensions.md)。

## 使用

Codex 插件为 `ariadne@ariadne-local`，技能 `$ariadne-reduction`；安装后刷新插件或新聊天以发现工具。

```text
使用 $ariadne-reduction 处理我指定的 Bruker/Rigaku 原始数据。
保留多个晶格假设与真实观测，评估扫描块留出和重叠可辨识性。
生成可信的 INS/HKL 后继续 solve-single-crystal。
```

`ariadne-env` 加载 Ariadne DIALS Python source overlay；`scxrd-env` 保留上游 DIALS 编译环境。Python 包、MCP namespace、skill、启动器、插件和本地 marketplace 均使用 Ariadne 名称。旧版源码和归档保留作历史版本。

## 安装

当前安装器面向 macOS/POSIX 环境，需要 Git、curl、patch、Python/conda 环境；Windows 原生部署尚未实现。

```bash
git clone https://github.com/owenxyzhang-seseTJ/Ariadne.git
cd Ariadne
sh scripts/install_local.sh
codex plugin marketplace add "$PWD" --json
codex plugin add ariadne@ariadne-local --json
```

安装器复用或安装 conda DIALS 3.30.1，下载匹配官方源码并应用 `patches/dials-3.30.1-ariadne.patch`，安装 Ariadne Python 包与官方 MCP SDK 1.30.0，配置独立启动器和 runtime.json。可传 base prefix/new runtime path，或设置 ARIADNE_MAMBA。GUI MCP 配置由安装器写当前主机绝对路径。没有修改 C++ 或重新编译完整 DIALS；没有安装商业还原套件。

## 实现和验证状态

已编写上述方法，并在开发主机配置环境和注册插件。只检查源码/API、安装元数据与注册结果。按用户要求，未执行算法、基准、模拟数据、真实数据或 MCP 连接测试，不报告准确率或提速收益。

完整多源 centroid 生成模型、正式的晶格数 model evidence、全局 composite-aware absorption/scaling、非负边界区间，以及 geometry/profile 误差传播仍未实现。校正工具传播真实外部校正因子，不能替代完整 TWINABS。Ariadne 保留原 signed-WLS 分支；NNLS 不自动替代弱峰的有符号测量。不同晶相不共用结构强度，也不能导成同一 HKLF5 孪晶模型。

新代码 MIT；DIALS 原源码/修改文件保留 BSD 许可。包中不含研究原始数据、商业程序或附件全文。部分 scientific schema 保留 scxrd-*/1 以兼容原有主数据；新增 Ariadne 方法对象使用 ariadne-*/1。

## 源码范围

本仓库包含 Ariadne 全部独立代码、Codex 插件和 DIALS 3.30.1 源码补丁。安装器从官方 tag 获取匹配上游源码后应用补丁，避免维护无修改的上游镜像。上游 BSD 许可保留在 `patches/DIALS-LICENSE`。不包含开发主机环境、研究样品、附件全文或商业程序。
