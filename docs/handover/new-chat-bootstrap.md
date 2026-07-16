# 新会话接管提示词

你可以直接复制以下内容到新的 AI Coding 会话作为初始系统提示：

---

你正在接管 YIAI物业 AI 智能客服与工单协同原型项目。

## 第一步：只读诊断

1. 读取仓库 `https://github.com/w497831176-rgb/greenview-ai-pm-demo` 的 Git main 最新代码。
2. 阅读以下两份交接文档（位于 `docs/handover/`）：
   - `project-context-v1.3.3.md`
   - `badcase-current-state-and-known-issues.md`
3. 在**不修改任何代码**的前提下，输出你对当前架构、Badcase 状态机、已知问题的理解摘要。
4. 不要立即写代码、不要立即部署、不要立即修复。

## 第二步：后续修复拆分

当你被明确授权开始修复时，按以下三批进行，每批完成后提交并推送：

### 第一批：前后端契约 + 状态机引导 + 草稿审核编辑

- 统一 `_enrich_badcase` 与前端字段契约，明确 `skill_prompt_drafts`、`capability_gap_drafts`、`retest_response`、`retest_context_json`、`badcase_actions` 的输出 schema。
- 前端根据 Badcase `status` 禁用不可用的操作按钮，避免用户点出 HTTP 400。
- 为 `knowledge_drafts`、`skill_prompt_drafts`、`capability_gap_drafts` 增加全文查看、编辑、审核页面或入口。
- 拆分"发布"与"应用"：草稿确认 → 预览 → 应用（记录版本）→ 自动触发重测。

### 第二批：Skill 应用绑定 + 路由修复草稿 + 复测证据

- 发布 Skill/Prompt 草稿时要求选择目标 Agent，并写入 `agent_skills` 绑定关系。
- 根据 Badcase `category` 展示分类化修复入口：知识缺口、Skill/Prompt 问题、MCP/能力缺口、路由问题、回答质量。
- 验证通过时要求操作员提供证据，不能仅检查 `retest_response` 是否存在。

### 第三批：自动监督模型与能力矩阵

- 扩展自动 Badcase 创建场景：低置信度路由、Skill 误触发、MCP 调用失败、模型回答质量异常等。
- 建立能力缺口矩阵，将 MCP 写操作、外部集成、数据缺失等分类沉淀到 `capability_gap_drafts`。

## 通用约束

- 不使用 `git push --force`，不使用 `git reset --hard`。
- 不删除业务会话、工单、知识库、Badcase 等现有数据。
- 不伪造验收结果；验收必须同时通过 API 脚本和前端页面（Playwright 或手动）。
- 不在任何文件、脚本、日志中记录密码、Token、API Key、SSH 凭证。
- 修改配置文件前必须先备份为 `.bak`。
- 每批修复完成后，更新 `docs/handover/badcase-current-state-and-known-issues.md` 中对应问题的状态。

## 当前关键上下文

- 官方 Agent：router、maintenance、billing、complaint、customer_service（共 5 个）。
- 官方 MCP Server：weather-server、calendar-server、workorder-server（共 3 个，均为只读）。
- 业主面聊天固定使用 `deepseek-v4-flash`；`deepseek-v4-pro` 仅用于 A/B 测试与 Darwin 深度分析。
- NAS 部署路径：`/volume3/docker/agno-demo-os`；前端 `http://192.168.50.123:18005`，API `http://192.168.50.123:8000`。
- 当前 main SHA：`e4768097b62a1a3e1abb47723211478092c1aae2`，Tag：`v1.3.3`。

---
