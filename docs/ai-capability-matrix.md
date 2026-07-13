# 绿景智服 V1.2｜AI 能力矩阵与演示证据计划

> 本文档用于面试准备和版本验收，明确每项能力的用户入口、涉及模块、技术原理、固定测试案例、演示证据和当前状态。  
> 状态说明：
> - `implemented_pending`：已实现，待环境修复后跑固定案例验证
> - `verified`：已通过固定案例验证
> - `next_version`：V1.3 或后续版本规划，当前未实现

---

## 1. Router 和垂直 Agent

| 项目 | 说明 |
|------|------|
| **用户入口** | 业主 Tab → AI 助手；平台管理 Tab → Agent 管理 |
| **涉及模块/API** | `agents/router.py`、`agents/maintenance.py` 等垂直 Agent、`app/agents.py`（`/api/agents`）、`app/chat.py`（`/api/chat/stream`） |
| **技术原理** | 用户消息先进入 Router，由模型输出 intent 与目标 Agent；Chat 模块将上下文交给对应垂直 Agent。垂直 Agent 绑定各自 Skill、知识与工具。 |
| **固定测试案例 ID** | I-01 ~ I-10（意图路由测试集） |
| **演示证据** | 发送「我家客厅吊灯不亮了」，SSE `event: route` 返回 `intent=maintenance`、`current_agent=维修 Agent`。 |
| **状态** | implemented_pending |

---

## 2. Skill

| 项目 | 说明 |
|------|------|
| **用户入口** | 平台管理 Tab → Skill 管理；业主 Tab → AI 助手（触发后气泡底部显示「已调用 Skill: xxx」） |
| **涉及模块/API** | `app/skills.py`（`/api/skills`）、`db/property_db.py`（`set_agent_skills`）、Agent 运行时上下文注入 |
| **技术原理** | Skill 是带触发条件与指令模板的能力单元。Agent 加载时读取启用状态的 Skill 并注入 Prompt；运行时根据关键词子串 + 二元组 Jaccard 触发。 |
| **固定测试案例 ID** | S-01 ~ S-03（Skill 触发测试）、B-01（词序反转 Badcase） |
| **演示证据** | 输入「帮我报修卫生间水管漏水」，AI 气泡底部显示「已调用 Skill: 维修工单处理」，并调用工单工具。 |
| **状态** | implemented_pending |

---

## 3. RAG（检索增强生成）

| 项目 | 说明 |
|------|------|
| **用户入口** | 业主 Tab → AI 助手；平台管理 Tab → 知识库管理 |
| **涉及模块/API** | `rag_retrieval.py`、`rag_indexer.py`、`rag_embeddings.py`、`app/knowledge.py`（`/api/knowledge/docs`、`/api/knowledge/search`）、`app/chat.py` |
| **技术原理** | 知识库文档按默认参数（chunk_size=512、chunk_overlap=64、split_strategy=auto）切分，经 BAAI/bge-small-zh-v1.5 Embedding 生成向量。用户提问时同步做关键词与向量检索，结果经 RRF 融合、重排后注入 Agent 上下文，并回显引用。 |
| **固定测试案例 ID** | R-01 ~ R-10（RAG 引用测试集） |
| **演示证据** | 问「物业费收费标准是什么」，AI 回答末尾或侧边显示引用卡片《物业服务收费标准》。 |
| **状态** | implemented_pending |

---

## 4. 混合检索与重排

| 项目 | 说明 |
|------|------|
| **用户入口** | 同上（RAG 调用链路内部） |
| **涉及模块/API** | `rag_retrieval.py`（`hybrid_search`、`keyword_search`、`semantic_search`、`rrf_fusion`、`rerank_chunks`） |
| **技术原理** | BM25 负责精确术语与数字命中；向量检索负责语义与同义改写；RRF 融合两套排序；Cross-Encoder 重排模型对 Top-K 再打分，取最优片段送入模型。 |
| **固定测试案例 ID** | R-03（装修押金）、R-05（宠物饲养）、R-08（水电费代缴） |
| **演示证据** | 平台管理 → 知识库 → 检索调试，输入「装修押金退还流程」，可见 BM25、向量、RRF 三列结果及最终重排序列。 |
| **状态** | implemented_pending |

---

## 5. 模型配置与 A/B

| 项目 | 说明 |
|------|------|
| **用户入口** | 平台管理 Tab → 模型配置；Skill 管理 → Skill 测试（选择对比模型） |
| **涉及模块/API** | `app/model_configs.py`（`/api/model-configs`）、`app/models_compat.py`（`/api/models/*`）、`app/settings.py`（`build_model`）、`app/skills.py`（Skill 测试） |
| **技术原理** | 默认生产模型为 DeepSeek V4 Flash（低成本、低延迟、开推理）；Pro 用于 A/B 对比与复杂分析。读取配置接口已脱敏 `api_key`，实际调用优先使用环境变量。 |
| **固定测试案例 ID** | M-01 ~ M-03（模型 A/B 测试集） |
| **演示证据** | 平台管理 → 模型配置，列表中不显示 api_key；Skill 测试时可选 Flash/Pro 对比输出。 |
| **状态** | implemented_pending |

---

## 6. MCP / Tool

| 项目 | 说明 |
|------|------|
| **用户入口** | 平台管理 Tab → MCP Server 管理；Agent 编辑页 → 绑定 MCP 工具 |
| **涉及模块/API** | `app/mcp.py`（`/api/mcp/servers`、`/api/mcp/tools`）、`tools/*_mcp_server.py`、Agent 运行时 MCPTools 绑定 |
| **技术原理** | MCP Server 以 stdio 方式启动本地子进程；Agent 通过 MCPTools 发现可用工具并在运行时调用。当前内置天气、日历、计算器、数据库查询等示例 Server。 |
| **固定测试案例 ID** | T-01 ~ T-03（MCP 工具调用测试集） |
| **演示证据** | 绑定天气工具后，业主问「今天适合晾衣服吗」，AI 调用 `get_weather` 并返回结果；Agent Trace 中可见 tool_call 节点。 |
| **状态** | implemented_pending |

---

## 7. Badcase 闭环

| 项目 | 说明 |
|------|------|
| **用户入口** | 业主 Tab → AI 助手消息旁 👎；平台管理 Tab → Badcase 库 |
| **涉及模块/API** | `app/badcases.py`（`/api/badcases`）、`db/property_db.py`、知识库/Skill/Prompt 管理 |
| **技术原理** | 用户点踩后生成 Badcase 记录；管理员分析根因并选择优化动作（更新 Prompt、补充知识库、调整 Skill 触发）；修复后通过「复测」按钮重新跑同一 query 验证效果。 |
| **固定测试案例 ID** | B-01 ~ B-03（Badcase 优化前后对比） |
| **演示证据** | 点踩 → 平台管理 Badcase 库新增记录 → 填写根因与修复动作 → 点击复测 → 同一 query 输出改善。 |
| **状态** | implemented_pending |

---

## 8. 人机协同与状态持久化

| 项目 | 说明 |
|------|------|
| **用户入口** | 员工 Tab → 待办工单 / 工单处理；业主 Tab → 我的工单 |
| **涉及模块/API** | `app/work_orders.py`（`/api/work-orders`）、`app/chat.py`（人工接管）、`db/property_db.py` |
| **技术原理** | 工单状态机：待处理 → 处理中 → 已完成/已关闭。员工可「接管」暂停 AI 自动回复，人工处理后更新状态；所有对话、工单、Badcase 持久化到 SQLite/Postgres，页面刷新不丢失。 |
| **固定测试案例 ID** | H-01 ~ H-03（人机协同测试集） |
| **演示证据** | 业主报修后生成工单 → 员工 Tab 可见待办 → 点击接管状态变「处理中」 → 更新备注 → 标记完成 → 业主 Tab「我的工单」状态同步。 |
| **状态** | implemented_pending |

---

## 9. 多模态报修

| 项目 | 说明 |
|------|------|
| **用户入口** | 业主 Tab → AI 助手 → 图片上传按钮 |
| **涉及模块/API** | 前端图片上传组件、`app/chat.py`（文件接收与多模态消息）、YIAI/Dify 多模态 Chatflow、工单创建工具 |
| **技术原理** | 业主上传图片后，系统调用多模态模型理解图片内容，提取故障类型、位置、严重程度，预填工单字段。 |
| **固定测试案例 ID** | V-01 ~ V-03（多模态报修测试集） |
| **演示证据** | 上传一张「吊灯不亮」照片 → AI 返回「检测到客厅主灯不亮，可能为灯泡/线路故障，已预填工单」 → 用户确认后创建工单。 |
| **状态** | next_version |

---

## 汇总表

| 能力 | 状态 | 关键证据 | 阻塞/备注 |
|------|------|---------|----------|
| Router 和垂直 Agent | implemented_pending | SSE route 事件 | 模型调用超时待修复 |
| Skill | implemented_pending | 气泡底部 Skill 标签 | 模型调用超时待修复 |
| RAG | implemented_pending | 引用卡片 | 模型调用超时待修复 |
| 混合检索与重排 | implemented_pending | 检索调试三列结果 | 模型调用超时待修复 |
| 模型配置与 A/B | implemented_pending | 配置接口脱敏、A/B 对比 | 模型调用超时待修复 |
| MCP / Tool | implemented_pending | tool_call 节点 | 模型调用超时待修复；当前仅 stdio 方式 |
| Badcase 闭环 | implemented_pending | Badcase 库 + 复测按钮 | 模型调用超时待修复 |
| 人机协同与状态持久化 | implemented_pending | 工单状态流转 | 模型调用超时待修复 |
| 多模态报修 | next_version | — | V1.3 规划 |

---

## 面试表述建议

> "V1.2 已经实现了前 8 项能力，覆盖了物业 AI 助手所需的核心 Agent、Skill、RAG、MCP、Badcase 和工单协同；多模态报修明确放到 V1.3。当前唯一阻塞是测试环境的模型调用超时，修复后跑一遍固定测试集，8 项能力均可验证。"
