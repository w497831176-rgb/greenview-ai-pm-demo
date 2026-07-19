# ADR-0004：配置驱动 ToolPlan、检索前 RAG Scope 与显式 Acceptance Runner

- 状态：Accepted
- 日期：2026-07-19
- 范围：V1.8 Goals 1–2
- 替代：V1.7 运行时中的 server-name MCP 规划与全库 Top-K 后过滤路径

## 背景

V1.8 的目标不是再增加物业领域分支，而是证明平台新增的垂直 Agent、Skill、
MCP 和 RAG 能在发布后的下一新会话中真实参与运行。此前仍有三处权力漂移：

1. read MCP 的稳定预调用按内置 server name 写分支；
2. write MCP 要求用户说出内部 tool name；
3. RAG 先做全库 Top-K，再过滤 Agent 绑定文档。

这些行为会让物业内置案例稳定，却无法排除“核心运行时写死业务 if/else”。

## 决策

### 1. Tool Metadata 是 ToolPlan 的唯一自然语言配置源

MCP discovery 只证明工具存在，不自动授予运行权限。操作者必须在 Draft 中确认：

- `effect`
- `risk_level`
- `natural_language_intents`
- `trigger_keywords` / `trigger_mode`
- `execution_mode`
- `argument_bindings`
- `result_contract`

名称前缀不再成为发布后的 ToolEffect 授权事实。未分类 Tool 进入快照时保持禁用。
V1.7 内置演示工具通过只读的兼容元数据进入相同编译流程；平台配置优先，不在
启动时回写或覆盖数据库。

### 2. ToolPlan 必须先于调用产生

运行时从一个不可变 Session Snapshot 读取 Agent-MCP 绑定、ToolPolicy 和 Tool
Metadata，生成结构化 `ToolPlan`。ToolPlan 包含匹配理由、参数、缺失字段、
JSON Schema 错误、effect、执行模式和结果契约。

- read Tool：仅在白名单内自动预调用或进入 Agno 原生工具循环；
- create/update Tool：只生成 Proposal；
- delete/unknown Tool：拒绝；
- 歧义或同分多候选：拒绝并要求澄清；
- 确认后的写调用仍需再次执行 Snapshot Schema 与 ToolPolicy 校验；
- 只有结果契约认可的业务成功状态、真实资源 ID 和 committed Receipt 才能宣称成功。

### 3. Agent RAG Scope 在每个召回通道的 Top-K 前生效

`allowed_document_ids` 是检索合同的一部分：

- keyword：构建检索索引时过滤；
- semantic：pgvector SQL 的 `WHERE` 条件过滤；
- RRF/rerank：只接收范围内候选；
- `None` 仅用于平台全局调试；
- 空数组代表该 Agent 无 RAG，不继承全库文档。

Citation、UI 点击分片和 Trace 继续由同一 EvidenceSet 生成。

### 4. 组合验收是显式 Runner，不是普通聊天的隐藏特权

普通业主聊天仍只能进入咨询、Handoff 或受控写路径之一。组合 Acceptance Run
只读取并聚合多个已经独立执行的子 Trace，不扩大任一子 Trace 的权限。

Acceptance Runner 提供：

- 单 Trace 命中核对；
- 新能力包的无模型 Snapshot/Router/Skill/ToolPlan/RAG Scope 核对；
- 多子 Trace 的父 Acceptance Run；
- Release、Snapshot、Trace、Evaluation、Badcase、Cost 和清理策略关联。

## 后果

正面结果：

- 新领域 MCP 不需要修改核心运行时代码；
- 用户不需要知道内部函数名；
- 新增离题 RAG 不会被物业全库候选挤出 Top-K；
- “代码存在”和“真实命中”可用机器证据区分；
- 普通咨询不会为了全栈演示获得写权限。

代价与限制：

- discovery 后需要操作者补齐治理元数据；
- 自动参数抽取只支持明确的声明式规则；复杂参数继续由模型原生工具选择或要求用户补充；
- 真实模型回答、真实 MCP 服务和 UI 体验仍需定向后端验收与用户手点，不能由无模型合同代替。

## 回滚

代码可回滚到上一 RuntimeRelease/应用 SHA；已有 Session Snapshot 不热切换。
回滚不得删除历史 Acceptance Run、Trace、Receipt 或用户业务数据。
