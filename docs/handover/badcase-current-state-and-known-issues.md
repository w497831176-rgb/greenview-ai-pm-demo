# Badcase 当前状态与已知问题

> 本文档以 GitHub main 当前代码事实为准，只记录、不修复。后续修复请按 `new-chat-bootstrap.md` 中的批次拆分。

## 1. Badcase 状态机

### 1.1 合法状态

```
pending → classified → fixing → verifying → closed
                     ↘                  ↗
                        → rejected
```

- `pending`：新建，未分类。
- `classified`：已分类（人工或 Darwin）。
- `fixing`：正在生成/发布草稿。
- `verifying`：已发布修复并执行真实重测，等待人工验证。
- `closed`：验证通过。
- `rejected`：被驳回（需填写 `rejected_reason`）。

### 1.2 各 API 动作的前置状态与副作用

| API | 前置状态要求 | 状态转移 | 副作用 |
|-----|--------------|----------|--------|
| `POST /api/badcases/{id}/classify` | `pending` 或 `classified` | → `classified` | 写入 category / root_cause / priority |
| `POST /api/badcases/{id}/extract-knowledge` | 无硬性检查 | `classified` → `fixing` | 创建 `knowledge_drafts` 记录 |
| `POST /api/badcases/{id}/publish-draft/{draft_id}` | 无硬性检查 | `fixing` → `verifying` | 创建正式知识库文档并 reindex |
| `POST /api/badcases/{id}/publish-skill-draft/{draft_id}` | 无硬性检查 | `fixing` → `verifying` | 更新或新建 Skill |
| `POST /api/badcases/{id}/accept-capability-gap/{draft_id}` | 无硬性检查 | 保持原状态 | 仅更新草稿状态为 accepted，记录待办 |
| `POST /api/badcases/{id}/darwin-fix` | 必须为 `classified` | → `fixing` | 调用 deepseek-v4-pro，生成三类草稿，记录独立 darwin trace |
| `POST /api/badcases/{id}/darwin` | 同 darwin-fix（前端别名） | 同 darwin-fix | 同 darwin-fix |
| `POST /api/badcases/{id}/darwin-optimize` | 同 darwin-fix（测试别名） | 同 darwin-fix | 同 darwin-fix |
| `POST /api/badcases/{id}/switch-model-retry` | 无硬性检查 | `pending/classified` → `fixing` | 用指定模型（默认 Flash）重跑问题，不进入真实 chat runtime |
| `POST /api/badcases/{id}/retest` | 必须为 `fixing` 或 `verifying` | → `verifying` | 通过真实 chat runtime（`_stream_agent_response`）重跑原始问题 |
| `POST /api/badcases/{id}/verify` | 通过：必须为 `verifying` 且存在 `retest_response`；不通过：无前置 | `verifying` → `closed` 或 `verifying` → `fixing` | 记录验证结果 |
| `POST /api/badcases/{id}/close` | 同 verify passed | → `closed` | 实际调用 verify_badcase(passed=True) |
| `POST /api/badcases/{id}/reject` | 无硬性检查 | → `rejected` | 必须提供 rejected_reason |
| `POST /api/badcases/{id}/transition` | 无硬性检查 | 任意合法状态之间 | 人工强转，记录 action |
| `POST /api/badcases/{id}/check-tools` | 无硬性检查 | `pending/classified` → `fixing` | 分析是否由工具缺失导致 |

### 1.3 状态机缺口

- 多个发布/接受动作没有前置状态检查，前端在错误状态下点击会收到 HTTP 400。
- `verify` 通过仅检查 `retest_response` 是否存在，不检查回答内容质量。
- `transition` 允许任意合法状态跳转，可能绕过真实修复流程。

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

## 3. 当前已知问题（按建议修复顺序）

### a. Darwin / 复测按钮缺少状态引导，用户可点出 HTTP 400

- **事实**：前端 Badcase 详情页展示了"Darwin 分析"、"复测"、"验证通过"等按钮，但未根据当前 `status` 禁用不可用的动作。
- **后果**：操作员可能在 `pending` 状态点击"复测"，或在 `fixing` 状态点击"验证通过"，后端返回 `400 cannot retest from status ...` 或 `must retest before verify`。
- **修复方向**：前端根据 `status` 动态控制按钮可用性；或后端统一返回更友好的错误，并在详情中暴露 `allowed_actions` 列表。
- **优先级**：第一批。

### b. 所有分类都显示"提取知识草稿"，缺少分类化修复入口

- **事实**：`extract-knowledge` 接口对所有 category 都可用，前端目前倾向于展示"提取知识草稿"入口。
- **后果**：`mcp_capability`、`routing`、`response_quality` 等分类无法直接生成对应的 skill_prompt 或 capability_gap 草稿，操作员只能走知识库路径。
- **修复方向**：根据 `category` 展示不同修复入口：`knowledge_gap` → 知识草稿；`skill_prompt` / `routing` → Skill/Prompt 草稿；`mcp_capability` → 能力缺口草稿。
- **优先级**：第一批。

### c. 草稿不能全文查看、编辑、审核

- **事实**：`knowledge_drafts`、`skill_prompt_drafts`、`capability_gap_drafts` 表支持 status / content，但前端没有提供全文查看、编辑、审核确认页面。
- **后果**：操作员只能直接"发布"，无法在发布前核对模型生成的内容。
- **修复方向**：新增草稿详情/编辑页，支持 status 流转（draft → under_review → approved → published / rejected）。
- **优先级**：第一批。

### d. 知识场景"发布/应用"概念混淆

- **事实**：`publish-draft` 直接将 `knowledge_draft` 写入正式知识库并 reindex；`publish-skill-draft` 直接覆盖 Skill 内容。
- **后果**：没有"应用"的中间状态，发布后无法回滚到上一版本知识/Skill。
- **修复方向**：引入"确认应用"步骤：草稿 → 预览 → 应用（记录版本）→ 自动触发重测；知识库与 Skill 保留历史版本。
- **优先级**：第一批。

### e. 新 Skill 草稿发布后没有明确绑定目标 Agent

- **事实**：`publish-skill-draft` 在 `target_skill_id` 为空时会新建 Skill，但不会自动绑定到任何 Agent；已有 Skill 更新后绑定关系不变。
- **后果**：新建 Skill 即使发布也不会被任何 Vertical Agent 加载。
- **修复方向**：发布 Skill/Prompt 草稿时要求选择目标 Agent，并在 `agent_skills` 中建立绑定；或默认绑定到与 Badcase 关联的 Vertical Agent。
- **优先级**：第二批。

### f. 前后端字段契约不一致

- **事实**：
  - `skill_prompt_drafts` 表有 `skill_id`、`skill_name`、`title`、`prompt_content`、`trigger_keywords`；但 `_enrich_badcase` 与前端展示字段混用 `name` / `title` / `content`。
  - `capability_gap_drafts` 表有 `gap_type`、`suggested_action`；前端有时只展示 `description`。
  - `badcases` 表的 `retest_response`、`retest_context_json`、`retest_trace_id` 在前端被解析为 `retest_context`，但 `retest_response` 有时未在详情页展示。
  - `badcase_actions` 的 `action_detail` 是 JSON 字符串，前端未统一格式化展示操作历史。
- **后果**：详情页字段缺失、草稿列表展示混乱、操作历史可读性差。
- **修复方向**：统一 `_enrich_badcase` 输出 schema；前端按 schema 渲染；补充 API 契约文档或 Pydantic response model。
- **优先级**：第一批。

### g. 验证通过只检查有无复测回答，没有检查证据

- **事实**：`verify_badcase` 通过条件为 `case["status"] == "verifying" and case.get("retest_response")`。
- **后果**：即使复测回答仍然错误，也可被直接关闭。
- **修复方向**：验证时要求操作员勾选/填写证据项：RAG 命中、MCP 调用、路由正确、回答符合预期；或增加自动检查项（如复测 Trace 的 category 是否仍为同一类问题）。
- **优先级**：第二批。

### h. RAG 自动 Badcase 当前只覆盖零召回，未覆盖路由/Skill/MCP 监督

- **事实**：自动创建 Badcase 的唯一入口是 `citations` 为空。
- **后果**：路由错误、Skill 误触发、MCP 调用失败、回答质量差等场景不会自动建单。
- **修复方向**：扩展自动监督模型：路由置信度低、Skill 触发后未产生有效结果、MCP 返回错误、模型输出包含"无法""超出范围"等模式时自动建单或提示运营。
- **优先级**：第三批。

## 4. 建议修复顺序

- **第一批**：前后端字段契约统一 + 状态机按钮引导 + 草稿查看/编辑/审核 + 发布/应用概念拆分。
- **第二批**：Skill 应用绑定 Agent + 分类化修复入口 + 复测证据检查。
- **第三批**：自动监督模型与能力矩阵（路由/Skill/MCP 监督）。
