# Badcase 当前状态与已知问题

> 本文档以 GitHub main 当前代码事实为准，记录已修复与仍待修复的问题。  
> **本版本**：`fix/v1.3.4-badcase-operator-closure`（第一批修复后）。  
> 后续修复请按本文档末尾的批次拆分继续推进。

## 1. Badcase 状态机

### 1.1 合法状态

```
pending → classified → fixing → verifying → closed
                     ↘                  ↗
                        → rejected
```

- `pending`：新建，未分类。
- `classified`：已分类（人工或 Darwin）。
- `fixing`：正在查看/编辑/审核草稿，或应用已审核草稿。
- `verifying`：已应用修复并执行真实重测，等待人工验证。
- `closed`：验证通过。
- `rejected`：被驳回（需填写 `rejected_reason`）。

### 1.2 各 API 动作的前置状态与副作用

| API | 前置状态要求 | 状态转移 | 副作用 |
|-----|--------------|----------|--------|
| `POST /api/badcases/{id}/classify` | `pending` | → `classified` | 写入 category / root_cause / priority |
| `POST /api/badcases/{id}/extract-knowledge` | `classified` 且 category=`knowledge_gap` | `classified` → `fixing` | 创建 `knowledge_drafts` 记录 |
| `POST /api/badcases/{id}/knowledge-drafts/{draft_id}/review` | `fixing` | 保持 `fixing` | 草稿状态 draft/under_review → approved/rejected |
| `PUT /api/badcases/{id}/knowledge-drafts/{draft_id}` | `fixing` 且草稿非 published/rejected | 保持 `fixing` | 更新草稿字段 |
| `POST /api/badcases/{id}/knowledge-drafts/{draft_id}/apply` | `fixing` 且草稿状态为 `approved` | → `verifying` | 创建正式知识库文档并 reindex；草稿状态变为 `published` |
| `POST /api/badcases/{id}/skill-prompt-drafts/{draft_id}/review` | `fixing` | 保持 `fixing` | 草稿状态流转 |
| `PUT /api/badcases/{id}/skill-prompt-drafts/{draft_id}` | `fixing` 且草稿非 published/rejected | 保持 `fixing` | 更新草稿字段 |
| `POST /api/badcases/{id}/skill-prompt-drafts/{draft_id}/apply` | `fixing` 且草稿状态为 `approved` | → `verifying` | 更新或新建 Skill；草稿状态变为 `published` |
| `POST /api/badcases/{id}/capability-gap-drafts/{draft_id}/review` | `fixing` | 保持 `fixing` | 草稿状态流转 |
| `PUT /api/badcases/{id}/capability-gap-drafts/{draft_id}` | `fixing` 且草稿非 accepted/rejected | 保持 `fixing` | 更新草稿字段 |
| `POST /api/badcases/{id}/capability-gap-drafts/{draft_id}/apply` | `fixing` 且草稿状态为 `approved` | → `verifying` | 草稿状态变为 `accepted`，仅记录产品待办，不创建真实工具 |
| `POST /api/badcases/{id}/publish-draft/{draft_id}` | 兼容别名，同 `apply` | 同 `apply` | 同 `apply` |
| `POST /api/badcases/{id}/publish-skill-draft/{draft_id}` | 兼容别名，同 `apply` | 同 `apply` | 同 `apply` |
| `POST /api/badcases/{id}/accept-capability-gap/{draft_id}` | 兼容别名，同 `apply` | 同 `apply` | 同 `apply` |
| `POST /api/badcases/{id}/darwin-fix` | 必须为 `classified` | → `fixing` | 调用 deepseek-v4-pro，生成三类草稿，记录独立 darwin trace |
| `POST /api/badcases/{id}/darwin` | 同 darwin-fix（前端别名） | 同 darwin-fix | 同 darwin-fix |
| `POST /api/badcases/{id}/darwin-optimize` | 同 darwin-fix（测试别名） | 同 darwin-fix | 同 darwin-fix |
| `POST /api/badcases/{id}/switch-model-retry` | 无硬性检查 | `pending/classified` → `fixing` | 用指定模型（默认 Flash）重跑问题，不进入真实 chat runtime |
| `POST /api/badcases/{id}/retest` | 必须为 `verifying` | → `verifying` | 通过真实 chat runtime（`_stream_agent_response`）重跑原始问题 |
| `POST /api/badcases/{id}/verify` | 通过：必须为 `verifying` 且存在 `retest_response`；不通过：必须为 `verifying` | `verifying` → `closed` 或 `verifying` → `fixing` | 不通过必须填写 note |
| `POST /api/badcases/{id}/close` | 同 verify passed | → `closed` | 实际调用 verify_badcase(passed=True) |
| `POST /api/badcases/{id}/reject` | 非 terminal 状态 | → `rejected` | 必须提供 rejected_reason |
| `POST /api/badcases/{id}/transition` | 状态机内合法转移 | 仅允许 1.1 图中边 | 人工强转，记录 action |
| `POST /api/badcases/{id}/check-tools` | `pending` 或 `classified` | 保持原状态 | 分析是否由工具缺失导致（后端诊断，不在前端操作按钮中列出） |

### 1.3 状态机缺口（第一批修复后）

- `transition` 已服从状态机图，不能绕过。
- 各动作已增加前置状态检查，前端根据 `allowed_actions` 隐藏/禁用不可用按钮。
- `verify` 通过仍只检查 `retest_response` 是否存在，未检查回答内容质量（第二批处理）。

## 2. 手动点踩与 RAG 零召回自动建单的区别

| 维度 | 手动点踩（source=manual） | RAG 零召回自动建单（source=auto） |
|------|---------------------------|-----------------------------------|
| 触发位置 | `app/chat.py:chat_feedback` | `app/chat.py:_stream_agent_response` 检索阶段 |
| 触发条件 | 业主点击 thumb_down 并填写原因 | `citations` 为空 |
| 初始 category | `pending` | `knowledge_gap` |
| 初始 status | `pending` | `pending`（代码中创建后无额外状态变更） |
| 保存字段 | title / description / source_message_id / session_id / source / original_query / ai_response / feedback_reason / context_json / trace_id / priority / message_id | title / description / category / evidence / source_message_id / session_id |
| 是否包含完整 Trace | 是 | 否 |
| 是否自动分类 | 否，需人工或 classify 接口 | 已预置为 knowledge_gap |
| 覆盖场景 | 任意回答质量、能力、路由、体验问题 | 仅覆盖知识库未命中 |

## 3. 已知问题（按建议修复顺序）

### a. Darwin / 复测按钮缺少状态引导，用户可点出 HTTP 400

- **状态**：第一批已修复。
- **实现**：
  - 后端 `app/badcase_schema.py` 统一定义状态机与 `allowed_actions`。
  - 后端各 API 使用 `_require_case_status` 强制校验前置状态。
  - 前端详情页根据 `bc.allowed_actions` 与 `bc.status` 动态展示/禁用按钮，terminal 状态只读。
- **验收**：`tests/test_badcase_schema.py` 覆盖状态机；`scripts/test_v1_3_4_badcase_operator_closure.py` 覆盖 API 行为。

### b. 所有分类都显示"提取知识草稿"，缺少分类化修复入口

- **状态**：第一批已修复。
- **实现**：
  - `knowledge_gap`：显示"提取知识草稿"，调用 `extract-knowledge`。
  - `skill_prompt` / `routing` / `response_quality`：显示"生成 Skill/Prompt 草稿"或"生成修复计划草稿"，调用 `darwin-fix`。
  - `mcp_capability`：显示"记录能力缺口草稿"，调用 `darwin-fix`。
  - 后端 `extract-knowledge` 限制 category 为 `knowledge_gap`。

### c. 草稿不能全文查看、编辑、审核

- **状态**：第一批已修复。
- **实现**：
  - 为 `knowledge_drafts`、`skill_prompt_drafts`、`capability_gap_drafts` 新增 `/{case_id}/...-drafts/{draft_id}`（编辑）、`/review`（审核）、`/apply`（应用）端点。
  - 前端草稿卡片提供"查看全文"、"编辑"、"通过审核/退回草稿/驳回"、"应用已审核草稿"操作。
  - 草稿状态支持 `draft` → `under_review` → `approved` → `published` / `rejected`。

### d. 知识场景"发布/应用"概念混淆

- **状态**：第一批已修复。
- **实现**：
  - 旧 `publish-draft`、`publish-skill-draft`、`accept-capability-gap` 变为兼容别名，要求草稿状态为 `approved` 且 Badcase 状态为 `fixing`。
  - 新端点统一为 `/apply`；应用成功后 Badcase 进入 `verifying`。
  - 前端按钮文案统一为"应用已审核草稿"，应用前二次确认。

### e. 新 Skill 草稿发布后没有明确绑定目标 Agent

- **状态**：仍待修复（第二批）。
- **说明**：本批 Skill 草稿 `/apply` 仅更新/新建 Skill，尚未写入 `agent_skills` 绑定关系。前端已预留目标 Skill ID 输入框，运行时绑定验证属于第二批。

### f. 前后端字段契约不一致

- **状态**：第一批已修复。
- **实现**：
  - 新增 `app/badcase_schema.py` 作为前后端字段契约单一来源。
  - `GET /api/badcases/{id}` 稳定返回：`query`、`category`、`category_label`、`source`、`source_label`、`context`、`root_cause`、`fix_plan`、`darwin_analysis`、`evidence`、`retest_response`、`retest_context`、`retest_trace_id`、`actions`、`knowledge_drafts`、`skill_prompt_drafts`、`capability_gap_drafts`、`allowed_actions`、`is_terminal`。
  - 前端清理 `retest_result`、`skill_drafts`、`capability_drafts`、`gap_drafts` 读取。
  - 操作历史统一格式化展示 `action_type`、`status_before/after_label`、`action_detail_parsed`、时间。

### g. 验证通过只检查有无复测回答，没有检查证据

- **状态**：仍待修复（第二批）。
- **说明**：`verify_badcase(passed=True)` 当前仍仅要求 `retest_response` 非空，未强制要求操作员提供质量证据或自动检查 RAG/MCP/路由正确性。

### h. RAG 自动 Badcase 当前只覆盖零召回，未覆盖路由/Skill/MCP 监督

- **状态**：仍待修复（第三批）。
- **说明**：自动创建 Badcase 的唯一入口仍是 `citations` 为空。路由置信度低、Skill 误触发、MCP 调用失败、模型回答质量差等场景尚未自动建单。

## 4. 建议修复顺序

- **第一批**（已修复）：前后端字段契约统一 + 状态机按钮引导 + 草稿查看/编辑/审核 + 发布/应用概念拆分。
- **第二批**（待修复）：Skill 应用绑定 Agent + 分类化修复入口完善 + 复测证据检查。
- **第三批**（待修复）：自动监督模型与能力矩阵（路由/Skill/MCP 监督）。
