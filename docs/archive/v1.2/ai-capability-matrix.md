# YIAI物业 V1.2｜AI 能力矩阵与演示证据计划

> 历史说明：本文保留 V1.2 能力演进记录。当前面试范围以 V1.8 四目标文档和
> ADR-0005 为准；图片上传、Vision/OCR 与多模态报修已主动退役，不再规划或演示。
>
> 本文档用于面试准备和版本验收，明确每项能力的用户入口、涉及模块、技术原理、固定测试案例、演示证据和当前状态。  
> 状态说明：
> - `implemented_pending`：已实现，待环境修复后跑固定案例验证
> - `verified`：已通过固定案例验证
> - `next_version`：V1.3 或后续版本规划，当前未实现

---

## 模型分层总览

| 模型 | 角色 | 当前代码位置 | V1.2 状态 |
|------|------|------------|----------|
| `deepseek-v4-flash` | 默认文本 Router 与常规垂直 Agent 主力模型；低成本、低延迟、开 reasoning | `app/settings.py` 默认 `MODEL`；`app/model_configs.py` 默认配置 | 已接入 |
| `deepseek-v4-pro` | 复杂文本任务、Badcase/Darwin 深度分析、Flash/Pro A/B 对比 | `app/model_configs.py` `/ab-test` 默认 `model_b`；`badcases.py` 的 `switch-model-retry` | 已接入 |

**关键约束**：
- 当前业主聊天只接受文本输入。
- 模型分层、Token 与成本治理只覆盖四目标所需的文本运行链。

---

## 1. Router 和垂直 Agent

| 项目 | 说明 |
|------|------|
| **用户入口** | 业主 Tab → AI 助手；平台管理 Tab → Agent 管理 |
| **涉及模块/API** | `agents/router.py`（`classify_intent`）、`agents/maintenance.py`（`create_maintenance_agent`）、`agents/billing.py`、`agents/complaint.py`、`agents/customer_service.py`、`agents/property.py`、`app/chat.py`（`/api/chat/stream`） |
| **技术原理** | `classify_intent` 调用 DeepSeek Flash 输出 JSON `{intent, reason}`，可选值：`maintenance`、`billing`、`complaint`、`customer_service`、`other`。`chat.py` 的 `_select_agent` 根据 intent 选择对应垂直 Agent 工厂函数。 |
| **固定测试案例 ID** | I-01 ~ I-05（覆盖 maintenance、billing、complaint、customer_service、other） |
| **演示证据** | 发送「我家客厅吊灯不亮了」，SSE `event: route` 返回 `{"intent": "maintenance", "current_agent": "维修 Agent", "reason": "用户描述家庭照明设备故障"}`。 |
| **状态** | implemented_pending |

---

## 2. Skill

| 项目 | 说明 |
|------|------|
| **用户入口** | 平台管理 Tab → Skill 管理（新建/编辑/导入/导出）；业主 Tab → AI 助手（触发后气泡底部显示「已调用 Skill: xxx」） |
| **涉及模块/API** | `app/skills.py`（`/api/skills`、POST/PUT/DELETE、导入/导出/测试）、`skill_storage.py`（SKILL.md 解析与存储）、`db/property_db.py`（`set_agent_skills`）、`app/chat.py`（`_build_skill_context`） |
| **技术原理** | Skill 以 YAML frontmatter + Markdown body 的 SKILL.md 形式持久化，frontmatter 支持 `version`、`name`、`trigger_condition` 等元数据。`chat.py` 加载启用状态的 Skill，按触发条件筛选；命中者注入 Agent Prompt 并显示 Skill 标签。触发策略为「关键词子串命中」或「字符二元组 Jaccard ≥ 0.45」。Skill 可单独指定 `model_id` 覆盖默认模型。 |
| **固定测试案例 ID** | S-01 ~ S-06 |
| **演示证据** | 平台管理 → Skill 管理 → 新建 Skill「维修工单处理」→ 填写触发条件「报修、查询工单、维修进度」与指令模板 → 保存后编辑 Agent 绑定该 Skill → 业主发送「帮我报修水管漏水」，AI 气泡底部显示 `已调用 Skill: 维修工单处理`。 |
| **已知边界** | 当前 **无自动版本历史/回滚**；修改 SKILL.md 即覆盖，需自行导出备份。 |
| **状态** | implemented_pending |

---

## 3. RAG（检索增强生成）

| 项目 | 说明 |
|------|------|
| **用户入口** | 业主 Tab → AI 助手；平台管理 Tab → 知识库管理 → 文档管理 / 检索调试 |
| **涉及模块/API** | `rag_indexer.py`（文档切分与索引）、`rag_embeddings.py`（BAAI/bge-small-zh-v1.5 Embedding）、`rag_retrieval.py`（检索与重排）、`app/knowledge.py`（`/api/knowledge/docs`、`/api/knowledge/search`、`/api/knowledge/{id}/chunks`）、`app/chat.py`（`_build_rag_context`） |
| **技术原理** | 上传/编辑文档后，系统按默认参数 `chunk_size=512`、`chunk_overlap=64`、`split_strategy="auto"` 切分，生成向量索引并持久化。用户提问时，`chat.py` 调用 `rag_retrieval.advanced_search` 获取 Top-K 片段并注入上下文；回答中通过 `citations` 回显引用。 |
| **固定测试案例 ID** | R-01 ~ R-06 |
| **演示证据** | 平台管理 → 知识库 → 上传《物业服务收费标准》PDF/TXT → 查看文档 chunks → 业主问「物业费收费标准是什么」→ AI 回答末尾显示引用卡片《物业服务收费标准》。 |
| **状态** | implemented_pending |

---

## 4. 混合检索与重排

| 项目 | 说明 |
|------|------|
| **用户入口** | 平台管理 Tab → 知识库管理 → 检索调试 |
| **涉及模块/API** | `rag_retrieval.py`（`keyword_search`、`semantic_search`、`rrf_fusion`、`rerank_chunks`、`advanced_search`）、`app/knowledge.py`（`/api/knowledge/search`） |
| **技术原理** | `advanced_search` 内部先分别执行 BM25 和向量语义检索，再用 RRF（Reciprocal Rank Fusion）融合两套排序；若开启 `enable_rerank`，则用 Cross-Encoder 对 Top-K 片段二次打分。最终按阈值过滤后返回。 |
| **各环节解决的问题** | **BM25**：精确术语、房间号、费用数字命中；**向量检索**：同义改写、语义相似；**RRF 融合**：兼顾召回与语义相关性，避免单一策略偏见；**阈值过滤**：剔除低置信片段；**重排**：提升最相关片段排名。 |
| **固定测试案例 ID** | H-01 ~ H-03 |
| **演示证据** | 平台管理 → 知识库 → 检索调试，输入「装修押金退还流程」→ 调试面板展示 keyword、semantic、advanced 三列结果及各自 score；advanced 列含 RRF 重排后的最终片段。 |
| **状态** | implemented_pending |

---

## 5. 模型配置与 A/B

| 项目 | 说明 |
|------|------|
| **用户入口** | 平台管理 Tab → 模型配置；Skill 管理 → Skill 测试 |
| **涉及模块/API** | `app/model_configs.py`（`/api/model-configs`、`/api/model-configs/ab-test`）、`app/settings.py`（`build_model`）、`app/models_compat.py`、Skill 测试接口 |
| **技术原理** | `model_configs` 表存储模型 ID、名称、provider、base_url、model_params 等；`build_model(model_id)` 按配置构造 DeepSeek 实例；读取端点通过 `_sanitize_config` 移除 `api_key`。A/B 接口 `POST /api/model-configs/ab-test` 并发调用 `model_a` 与 `model_b`（默认 Flash/Pro），返回两条 response。 |
| **固定测试案例 ID** | M-01 ~ M-03 |
| **演示证据** | 平台管理 → 模型配置，列表中 `api_key` 为空；POST `/api/model-configs/ab-test` 同一 prompt 返回 Flash 与 Pro 两份输出；人工对比质量并记录评分。 |
| **重要边界** | `/ab-test` 接口**不自动返回延迟和成本**；延迟需客户端计时，成本需按各自 token 用量 × 单价人工估算。 |
| **状态** | implemented_pending |

---

## 6. MCP / Tool

| 项目 | 说明 |
|------|------|
| **用户入口** | 平台管理 Tab → MCP Server 管理；Agent 编辑页 → 绑定 MCP 工具 |
| **涉及模块/API** | `app/mcp.py`（`/api/mcp/servers`、`/api/mcp/tools`、`/api/mcp/servers/{id}/discover`）、`app/chat.py`（`_build_mcp_tools`、`_format_mcp_context`） |
| **技术原理** | MCP Server 以 stdio 方式启动本地子进程；`MCPTools` 发现可用工具并在运行时调用。`chat.py` 将启用的 Server 描述注入 Agent Prompt，要求 Agent 在相关问题中必须调用工具而不得猜测。 |
| **固定测试案例 ID** | T-01 ~ T-03 |
| **演示证据** | 平台管理 → MCP Server 管理 → 新增天气 Server（命令 + 参数）→ 点击 discover 显示 `get_weather` → 编辑维修 Agent 绑定该工具 → 业主问「今天适合晾衣服吗」→ SSE 中 `event: tool_calls` 显示 `get_weather` 调用，最终回答中回注天气结果。 |
| **状态** | implemented_pending |

---

## 7. Badcase 闭环

| 项目 | 说明 |
|------|------|
| **用户入口** | 业主 Tab → AI 助手消息旁 👎；平台管理 Tab → Badcase 库 |
| **涉及模块/API** | `app/badcases.py`（`/api/badcases`、`/{id}/classify`、`/{id}/extract-knowledge`、`/{id}/publish-draft/{draft_id}`、`/{id}/darwin-fix`、`/{id}/switch-model-retry`、`/{id}/retest`、`/{id}/verify`、`/{id}/check-tools`、`/{id}/actions`）、`db/property_db.py` |
| **真实能力名称** | `classify`、`extract-knowledge`、`publish-draft`、`darwin-fix`（别名 `darwin`、`darwin-optimize`）、`switch-model-retry`（别名 `retry`）、`retest`、`verify`（别名 `close`/`reject`）、`check-tools`、`transition`、`list-actions`。 |
| **生命周期** | `pending → classified → fixing → verifying → closed`（或 `rejected`）。 |
| **固定测试案例 ID** | B-01 ~ B-06 |
| **演示证据** | 点踩 → 创建 Badcase（pending）→ `/classify` 自动分类（classified）→ `/extract-knowledge` 生成知识草稿（fixing）→ `/publish-draft/{draft_id}` 发布为知识文档（verifying）→ `/retest` 用默认模型复测 → `/verify` 通过并关闭（closed）。 |
| **状态** | implemented_pending |

---

## 8. 人机协同与状态持久化

| 项目 | 说明 |
|------|------|
| **用户入口** | 员工 Tab → 待办工单 / 工单处理；业主 Tab → 我的工单；AI 助手中的「转人工」 |
| **涉及模块/API** | `app/work_orders.py`（`/api/work-orders`）、`app/chat.py`（`/api/chat/handoff`、`/api/chat/handoff-reply`、`/api/chat/handoff-resolve`、`/api/chat/handoffs`、`/api/chat/history`）、`db/property_db.py` |
| **技术原理** | 工单状态机：待处理 → 处理中 → 已完成/已关闭。`chat.py` 支持三种人工接管路径：业主触发关键词、AI 主动提出转人工、员工在列表中接管会话。接管期间 Agent 不再自动回复；员工回复以 `role=staff` 写入消息表；结束后 `resolve_handoff` 恢复 AI 自动回复。所有消息、工单、会话状态持久化到 SQLite/Postgres。 |
| **固定测试案例 ID** | C-01 ~ C-03 |
| **演示证据** | 业主说「我要人工客服」→ 会话状态变为 handoff_requested → 员工在「人工接管」列表点击接管 → 状态变为 active → 员工回复 → 刷新页面后聊天记录、引用、Skill 标签、工单状态均保留。 |
| **状态** | implemented_pending |

---

## 9. 已退役范围：多模态 Vision

| 项目 | 说明 |
|------|------|
| **决策** | ADR-0005：从当前产品、运行时和面试演示范围退役 |
| **已移除** | 图片上传/预览、Kimi Vision API、聊天图片上下文、相关环境变量 |
| **数据处理** | 历史表和资产非破坏性保留，但应用不再创建、读取或暴露 |
| **状态** | retired |

---

## 汇总表

| 能力 | 状态 | 关键证据 | 阻塞/备注 |
|------|------|---------|----------|
| Router 和垂直 Agent | implemented_pending | SSE route 事件 | 模型调用超时待修复 |
| Skill | implemented_pending | 气泡底部 Skill 标签 | 模型调用超时待修复；无自动版本历史/回滚 |
| RAG | implemented_pending | 引用卡片 | 模型调用超时待修复 |
| 混合检索与重排 | implemented_pending | 检索调试三列结果 | 模型调用超时待修复 |
| 模型配置与 A/B | implemented_pending | 配置接口脱敏、`/ab-test` 双模型输出 | 模型调用超时待修复；延迟/成本需人工记录 |
| MCP / Tool | implemented_pending | tool_calls 节点 | 模型调用超时待修复；当前仅 stdio 方式 |
| Badcase 闭环 | implemented_pending | Badcase 库 + 复测按钮 | 模型调用超时待修复 |
| 人机协同与状态持久化 | implemented_pending | 工单状态流转、handoff 会话 | 模型调用超时待修复 |
| 多模态 Vision | retired | ADR-0005 | 不属于当前四目标，不再演示 |

---

## 面试表述建议

> "当前演示聚焦四个可验证目标：能力实时命中、能力实时新增、能力优化闭环和能力成本治理。多模态 Vision 已主动退役，以减少与四目标无关的复杂度。"
