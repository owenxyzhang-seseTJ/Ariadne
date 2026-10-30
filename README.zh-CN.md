# Ariadne 0.7.0

面向单晶衍射的数值处理与原生 agent 工作流。软件连接数据约化、结构候选、精修、原生 CIF 检查和结构反馈，并保存任务依赖、计算产物与恢复记录。

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [许可证](LICENSE)

本版以恢复的 R2 源码为基线，重新实现 0.7.0 的结构反馈和 agent。仓库包含完整源码、DIALS 补丁、测试、示例、wheel、技术报告及本轮[源码审查记录](CODE_REVIEW.zh-CN.md)。

## 直接使用

完整操作说明见 [使用指南](AGENT_GUIDE.zh-CN.md)，原理和验证见 [技术报告 PDF](reports/Ariadne_0_7_0_Technical_Report.pdf)，可编辑原稿为同名 `.tex`。

```bash
export PYTHONPATH="$PWD/python/src${PYTHONPATH:+:$PYTHONPATH}"
python -m ariadne.agent.cli capabilities
python examples/agent/create_demo.py /absolute/path/to/NEW-DEMO
```

该示例需要 NumPy、SciPy、Gemmi，不需要 DIALS、SHELX、MCP 或在线大模型。它使用真实 Gemmi 计算合成结构的结构因子，运行五节点反馈工作流，检查暂停、恢复和不重复执行。

## 本版实际包含什么

保留 R2 的预测几何、体素轮廓、有符号 WLS、计数似然、缩放、导出和 DIALS 源码补丁。新增 CIF 原生检查、相关结构先验、局部块与低秩后验、训练专用校准、闭合数据划分、测试暴露记录，以及持久化任务图、真实候选展开和结构文件配对。

默认采用确定性计划器，也可接入支持结构化 JSON 的语言模型服务。模型只选择注册动作，不编造结果或直接执行 shell。13 类动作和六种工作模式由命令行查询；原 MCP 接口保留，新增八个会话工具。

## 已验证到哪里

推送前完整回归为 **323 项通过、11 项跳过、0 项失败或错误**，另有 136 个子测试通过。Gemmi 原生计算与 MCP 工具注册已实际执行；构建后的 wheel 已脱离源码目录安装并完成三节点 CIF 审查。当前记录见 [验证结果](reports/github_review_validation.json)。

跳过项依赖 DIALS/dxtbx 或 cctbx。SHELXT/SHELXL 和模型接口使用明确标识的协议测试程序，真实原始帧的端到端解析准确率尚未验证。技术报告中的旧测试数量对应初次重建，当前结果以本轮审查记录为准。

自动原始帧路线限单晶模型；复杂孪晶通过明确的 custom 联合处理图。结构反馈使用固定 Gaussian 观测模型，当前预测限实数 IT92 X 射线散射，不包含异常散射和溶剂掩膜。模型辅助后验不替代实验数据。

## 文件与许可

`RELEASE_MANIFEST.json` 保存大小和 SHA-256，使用 `python scripts/verify_release.py` 检查。`dist/` 提供 Python wheel；第三方运行库、SHELX 二进制和字体不在包内。原有许可证和依赖说明保留。

原 R2 指南位于 `docs/R2_README*`，旧报告和基准保留。重现本轮检查可运行 `python -m pip install -e './python[dev]'`，再运行 `python scripts/run_review_validation.py`。

原生 DIALS 路线使用 `sh scripts/install_local.sh` 创建匹配 DIALS 3.30.1 的独立 conda 环境与源码覆盖层。单独安装 pip cctbx 不能代替该运行环境。agent 支持 Linux、macOS 和 WSL。
