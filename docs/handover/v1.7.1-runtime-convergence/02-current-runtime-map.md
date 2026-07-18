# 02｜当前运行时地图与配置真相源

## 1. 当前主入口

核心入口是 `app/chat.py` 的 `_stream_agent_response()`。它目前在一个流式函数中串联了多种不同性质的事情：

```text
请求进入
  → 创建会话与 Trace、保存用户消息
  → 人机协同策略 / 已接管会话处理
  → 确定性工单工作流控制器
  → 从数据库读取动态垂直 Agent
  → Router 模型分类或能力回退
  → 选择 Agent
  → Skill 匹配与 Prompt 注入
  → RAG 检索与 citation
  → 内置只读 MCP 的策略预调用
  → 动态 MCP 的模型原生调用准备
  → 垂直 Agent 生成回答
  → Tool/RAG/Skill 证据核验、保存消息
  → Trace / model call / cost / 自动 Badcase
  → SSE 输出
```

这解释了项目复杂的直接原因：同一入口同时承担了**只读咨询、模型推理、受控写操作、审计记录、UI 事件协议**五类职责。

## 2. 当前控制器的职责边界

| 模块 | 当前职责 | 应保留的权威 |
| --- | --- | --- |
| `app/work_order_workflow.py` | 工单草稿、字段收集、确认、真实写库 | 是否写入工单只能由它决定 |
| Router / `app/agents.py` / `app/chat.py` | 从已启用垂直 Agent 中选择处理者 | 路由只决定“谁处理”，不直接写业务数据 |
| `app/skill_runtime.py` | Skill 触发、负向排除、优先级、冲突处理 | 是否命中 Skill 必须可解释、可复算 |
| `rag_retrieval.py` / `app/knowledge.py` | 分片、检索、融合、阈值、引用元数据 | 可回答的依据只能来自最终选中分片 |
| `app/mcp.py` / `app/mcp_contracts.py` | Server 发现、工具连接、调用审计 | 工具权限由当前 Agent 绑定决定 |
| `app/handoff*` / `app/chat.py` | 人机协同状态与接管包 | 人工接管后，AI 不能冒充人工做最终决定 |
| `app/observability.py` / `db/property_db.py` | Trace、model calls、价格、预算 | 展示必须使用实际记录和价格快照 |
| `app/badcases.py` / `app/evaluations.py` | 问题闭环与可重复评估 | 修复、复测、关闭必须有证据 |

## 3. 数据与配置的位置

### 3.1 物业业务数据

- 容器中业务数据目录：`/app/data`
- NAS 映射：`/volume3/docker/volumes/agno-demo-os/property-data`
- 物业业务事实表（Agent、Skill、绑定、知识文档、工单、聊天、Trace、价格、Badcase 等）主要来自 property SQLite 数据库。
- 本地 RAG 索引目录也位于 property-data 卷内。必须先以 `rag_retrieval.py` 和实际运行配置核验索引真相源。

### 3.2 Agent / Skill / MCP

- `agents`：可配置 Router / vertical Agent；
- `skills` + `agent_skills`：Skill 本身及绑定关系；
- `agent_tools`：Agent 可用工具配置；
- MCP Server 配置、发现的工具清单与实际调用审计：由对应表/API管理。

**重要风险：**运行时还存在代码中的官方 Agent factory、启动迁移与默认模型配置。新会话必须先做一张“配置真相源表”，明确每个字段以数据库、代码默认值还是环境变量为准；禁止让启动迁移悄悄覆盖用户在平台管理页做过的配置。

### 3.3 模型与成本

- 业主端 Router 与垂直 Agent：`deepseek-v4-flash`，thinking 开启；
- `deepseek-v4-pro`：A/B、Darwin 等高价值深度后台任务；
- Kimi Vision：图片理解/OCR 路径，不应成为默认文本模型；
- `model_prices`：价格快照；
- `model_calls` / `chat_traces` / trace events：每轮模型和链路审计。

成本字段必须区分：未缓存输入、缓存命中输入、输出、推理（如 Provider 提供）。Router 若只返回总 Token 而未拆分输入输出，应记录“usage split unavailable”，不能用猜测单价展示假精确成本。

## 4. 动态 Agent 的正确运行契约

动态 Agent 不应只出现在管理页。一个新 Agent 生效至少经过：

```text
创建 Agent（职责/描述/启用）
  → 绑定 Skill（正/负触发、版本、优先级）
  → 可选绑定 MCP Tool（最小权限）
  → Router 读取已启用 Agent 并选中它
  → Skill runtime 读取绑定关系并命中
  → 工具层只加载该 Agent 授权的工具
  → Trace 保存 route / skill / tool / model evidence
```

目前已验证 `儿童教育Agent + zhangxuefeng-perspective` 的确定性 Skill 匹配能命中“我家孩子不爱学习怎么办”。该验证并不代替一次真实模型回答验收；它证明的是“配置确实进入运行时”。

## 5. 当前最需要避免的耦合

1. 只读咨询因会话残留草稿被劫持到工单工作流；
2. 模型自然语言说“已创建”，却没有经过确定性写入控制器；
3. RAG 候选、模型文本中的引用编号、UI citation 卡片三者不是同一份结构化数据；
4. 同一内置 MCP 既被策略预调用又被模型原生重复调用；
5. 单个超长 Skill 无预算注入，撑大 Prompt 造成成本/超时/回答失控；
6. 价格配置有了，但 Provider usage 字段不完整，UI仍展示看似精确的成本。
