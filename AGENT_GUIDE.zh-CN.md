# Ariadne 0.7.0 使用指南

构建标识：`0.7.0-rebuild-20261030`。本版从能够恢复的 R2 完整包重新构建，保留其数值算法，新增结构反馈与原生 agent。它不是丢失的旧 0.7.0 压缩包的逐字节恢复件。

## 1. 先选择运行范围

| 任务 | 需要的运行环境 |
|---|---|
| CIF 检查、结构因子、固定 Gaussian 结构反馈、合成示例 | Python 3.12 或更新版本，NumPy、SciPy、Gemmi |
| 原始帧处理和 R2 原生适配器 | 匹配的 DIALS 3.30.1 / dxtbx / cctbx 环境及源码补丁 |
| 从 INS/HKL 解结构或精修 | 合法安装的 SHELXT / SHELXL 可执行文件 |
| MCP 客户端调用 | 额外安装 MCP Python SDK |
| 语言模型计划器 | 可选；默认确定性计划器不需要模型或 API key |

本轮实际测试环境为 Python 3.13.5、NumPy 2.3.5、SciPy 1.17.0、Gemmi 0.7.4。实际 DIALS、SHELXT、SHELXL、MCP 运行库和在线模型服务未在本轮运行。外部程序和模型接口使用明确标记的协议测试程序检验，不能理解为真实结构求解验证。

## 2. 安装和第一次运行

### 2.1 先运行不依赖 DIALS 的部分

在解压后的 Ariadne 根目录执行。建议新建虚拟环境，不要修改已有科学计算环境。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'numpy>=1.26' 'scipy>=1.11' 'gemmi>=0.7.4,<0.8' pytest
export PYTHONPATH="$PWD/python/src${PYTHONPATH:+:$PYTHONPATH}"
python -m ariadne.agent.cli capabilities
```

需要控制台入口和 MCP 时，可安装项目：

```bash
python -m pip install ./python
ariadne-agent capabilities
```

离线环境可使用自备的依赖 wheel；交付的 Ariadne wheel 不包含 NumPy、Gemmi、DIALS 或 SHELX。完整源码目录仍须保留，R2 脚本和专家知识包由 `ARIADNE_PLUGIN_ROOT` 定位。

```bash
export ARIADNE_PLUGIN_ROOT=/absolute/path/to/Ariadne
```

### 2.2 配置匹配的 DIALS 环境

已有 DIALS 3.30.1 conda 环境时，原安装器将源码覆盖层放入新的目录，保留编译扩展：

```bash
sh scripts/install_local.sh \
  /absolute/path/to/dials-conda-prefix \
  /absolute/path/to/NEW-ariadne-runtime
```

安装器会访问网络、下载对应源码并安装依赖；请先审阅脚本，使用新的 runtime 目录。它不是科学验证程序。启动器默认写入 `~/CrystallographyPrograms/ariadne-env`：

```bash
~/CrystallographyPrograms/ariadne-env python -m ariadne.agent.cli capabilities
```

不要把本轮 Python 3.13 的测试环境当成 DIALS 3.30.1 已验证的二进制环境。SHELXT、SHELXL 按各自许可单独安装；它们不在源码包内。

## 3. 一个可立即复现的完整示例

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/agent/create_demo.py /absolute/path/to/NEW-DEMO
```

脚本使用真实 Gemmi 计算两个合成 CIF 的结构因子，生成有弱分量和完全重叠组的 Gaussian 信息模式。它执行两个 CIF 检查、一次反馈循环、评审和打包，共五个任务；在第二项任务后暂停，再恢复完成，并确认再次恢复不重复执行。

生成的 `demo_summary.json` 给出真值、估计值、负观测个数、候选评分和恢复检查。数据完全合成；示例没有调用 SHELXT，也不是从原始实验帧解出了一个未知结构。

## 4. 六种任务入口

所有请求都使用 `schema: ariadne-agent/1`。相对输入路径相对于请求 JSON 所在目录解析；新会话目录必须不存在。

| mode | 输入 | 执行内容 |
|---|---|---|
| `audit` | `cif` | 实际读取 CIF，检查晶胞、对称性、元素、占有率和位移参数 |
| `solve` | `ins`、`hkl` | SHELXT → 实际候选配对 → SHELXL → 比较 → 评审 → 归档 |
| `refine` | `res`、`hkl` | 有限轮次 SHELXL 精修与产物审查 |
| `reduce` | 原始图像文件、组成 | DIALS 单晶路线 → SHELX 导出 → 解结构 |
| `feedback` | 信息模式、结构候选及来源 | 训练校准 → 验证选候选 → 冻结 → 一次最终测试 |
| `custom` | 明确的任务图 | 组合原有联合积分、缩放和反馈等注册动作 |

`examples/agent/` 提供 audit、solve、refine 请求模板。以解结构为例：

```json
{
  "schema": "ariadne-agent/1",
  "mode": "solve",
  "inputs": {
    "ins": "/absolute/path/to/sample.ins",
    "hkl": "/absolute/path/to/sample.hkl"
  },
  "settings": {"cycles": 8},
  "budget": {
    "max_actions": 40,
    "max_attempts": 2,
    "wall_seconds": 7200,
    "action_seconds": 600
  },
  "planner": {"kind": "deterministic"},
  "experts": ["crystallography", "mof", "twinning"]
}
```

```bash
python -m ariadne.agent.cli init solve.json sessions/example
python -m ariadne.agent.cli run sessions/example
python -m ariadne.agent.cli status sessions/example
python -m ariadne.agent.cli export sessions/example delivery.zip --include-inputs
```

执行程序由部署者配置：在项目请求之外保存 JSON，例如 `{"shelxt":"/absolute/path/to/shelxt","shelxl":"/absolute/path/to/shelxl"}`，并在启动 CLI/MCP 前设置 `ARIADNE_EXECUTABLES_CONFIG=/absolute/path/to/deployment.json`。请求中的非空 `executables` 被拒绝；模型不能更换执行程序。

原生 CIF 检查、结构预测和反馈计算在独立子进程中运行，整个进程组受动作期限与取消控制。暂停、恢复、输出和科学验收语义保持不变。

SHELXT 候选必须有实际产生的同名 RES/HKL。没有对应 HKL 的候选不会自动借用另一候选或最初输入的 HKL。程序退出成功也不等于结构可信；有效文件、原生解析和科学检查分别记录。

## 5. 从原始图像开始

```bash
python -m ariadne.agent.cli request-from-images \
  /absolute/path/to/images raw-request.json \
  --glob '*.cbf' --composition 'C20 H16 N4 O4 Cu'
python -m ariadne.agent.cli init raw-request.json sessions/raw
python -m ariadne.agent.cli run sessions/raw
```

上述组成仅为格式示例，必须换成实际组成。只选择读图后端支持的图像。HDF5 master/data、清单/NPZ 等有侧文件依赖的输入不能只复制主文件；应在 `custom` 中用 `files.unpack` 解包完整输入集合，保留目录布局，再显式引用对应文件。

内置 `reduce` 路线限定单晶模型。实际检查发现多个晶体模型时会阻止这条路线继续，不会用一个不合适的单晶导出掩盖复杂孪晶。复杂孪晶需使用明确的 `custom` 联合处理图；现有 R2 脚本全部保留。

## 6. 接入 R2 信息模式与结构反馈

结构反馈只接受实验来源的有符号 Gaussian 信息模式。原始 JSONL 到反馈模型的转换需要一份显式身份映射：每个贡献的实验域/HKL、唯一强度 ID、模型 HKL、已知校正乘数以及原始像素 ID 都要确定。

```bash
python -m ariadne.agent.cli bridge \
  integrated_groups.jsonl identity-map.json feedback-data.json
```

转换器检查原始强度约定，避免再次除以 partiality；同一潜在强度只施加一次先验。模型辅助后验和非负计数估计不会被自动伪装为独立 WLS 观测。未知的域映射、吸收或尺度约定不会由程序猜测。

`examples/agent/create_demo.py` 展示完整、可运行的 feedback 请求结构；`tests_rebuild/test_bridge.py` 提供最小身份映射示例。当前 Gemmi 路径使用实数 IT92 X 射线散射因子，不包含异常散射、未建模客体或溶剂掩膜；候选必须位于一致的晶胞与指标系。

候选来源须声明为外部来源或只使用训练观测获得。相同强度、声明的等价/孪晶关系属于同一划分组；原始像素重用会被拒绝。程序无法仅从一份 CIF 证明其训练来源，因此来源声明仍须真实。

验证只比较冻结观测模型下的预测分布。候选选定后，最终测试在计算前登记暴露。登记库位于会话父目录的 `.ariadne-feedback-ledger.sqlite`；保留它和会话一起归档。它是本地项目记录，不是跨机器、不可绕过的认证系统。

## 7. 暂停、取消和恢复

```bash
python -m ariadne.agent.cli run sessions/example --steps 2
python -m ariadne.agent.cli run sessions/example
python -m ariadne.agent.cli cancel sessions/example
python -m ariadne.agent.cli resume sessions/example
python -m ariadne.agent.cli run sessions/example
```

`resume` 清除取消标记，实际执行仍由 `run` 完成。已成功的节点不会重复计算。每次尝试使用独立目录；输入和产物在恢复前复核。

系统崩溃可能留下仍在运行的外部进程。程序不会自行宣称恰好执行一次；先检查进程与日志，确认停止后再恢复该节点：

```bash
python -m ariadne.agent.cli recover sessions/example NODE \
  --confirmed-process-stopped
python -m ariadne.agent.cli retry sessions/example NODE
```

反馈最终测试节点不允许用普通 retry 再次查看同一测试数据。失败不应通过删除记录来伪装为一次新的独立检验。

## 8. 可选语言模型计划器

默认运行不需要语言模型。启用时，将请求中的 planner 改为：

```json
{
  "kind": "chat-completions",
  "base_url": "https://YOUR-PROVIDER.example/v1",
  "model": "YOUR-MODEL",
  "api_key_env": "ARIADNE_MODEL_API_KEY",
  "timeout_seconds": 45
}
```

密钥在启动前由环境变量提供，不写入请求文件。模型只能选择就绪节点、暂停或建议一个注册动作；不能发送 shell、编造计算结果或绕过产物引用。它可以通过 `artifact.inspect` 读取限长文本或选定 JSON 字段，再据实际诊断提出分支。

本地服务只在显式设置 `allow_local_http: true` 时允许 loopback HTTP。在线请求会发送有限的状态和科学摘要，请先确认模型服务的数据政策。当前客户端面向通用 Chat Completions JSON 协议，不保证所有提供商的扩展行为一致。

## 9. MCP 和专家知识包

原有 MCP 工具保留，新增八个工具：`agent_capabilities`、`agent_init`、`agent_step`、`agent_status`、`agent_cancel`、`agent_resume`、`agent_retry`、`agent_export`。`agent_step` 同步执行一个受预算约束的动作，不是后台调度承诺。

```bash
python -m ariadne.server
```

`experts/` 中的 crystallography、mof、twinning 是声明式知识包，不是任意代码插件。增加可执行工具需要修改注册表、实现输入输出验证并加入测试。

## 10. 验证本次交付

```bash
python scripts/verify_release.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/run_rebuild_validation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/feedback_rebuild.py
```

GitHub 推送前的完整审查为 323 项通过、11 项跳过、0 项失败或错误，另有 136 个子测试通过。跳过项需要原生 DIALS/dxtbx 或 cctbx；Gemmi 和 MCP 已实际运行。当前检查见 `reports/github_review_validation.json`，可用 `python scripts/run_review_validation.py` 重现。

本轮修正见 `CODE_REVIEW.zh-CN.md`。原始帧快照保留共同目录和相对路径，反馈校准保留 R2 的晶畴编号，HKLF5 导出同时保存观测映射旁侧文件。最终测试登记使用原始观测身份，修改说明文字不会产生新的测试独立性。外部进程受超时控制；进程内数值计算在调用前后检查预算，正在执行的 BLAS 运算不能立即抢占。

旧报告和旧基准作为历史文件保留，不代表本次重新执行。当前完整技术报告为 `reports/Ariadne_0_7_0_Technical_Report.pdf`。文件校验清单只说明交付文件是否改变，不能替代科学验证。
