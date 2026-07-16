# YIAI物业 V1.3.3 项目交接上下文

> 本文档基于 GitHub main 当前代码事实编写，用于新会话或新开发者接管项目时快速定位边界。严禁在本仓库任何文件中记录密码、Token、API Key、SSH 凭证。

## 1. 版本与仓库信息

| 项目 | 值 |
|------|-----|
| 仓库地址 | `https://github.com/w497831176-rgb/greenview-ai-pm-demo` |
| 当前 main SHA | `e4768097b62a1a3e1abb47723211478092c1aae2` |
| 当前 Tag | `v1.3.3`（annotated tag 对象 `bd2258d29ff3dbbf0595da75824816c9bd253240`，指向 main SHA） |
| 上一版本 Tag | `v1.3.2`、`v1.3.1`、`v1.3.0`（未移动） |
| NAS 部署版本 | 容器内 `git rev-parse HEAD` 与 main SHA 一致 |
| NAS 访问入口 | 前端 `http://192.168.50.123:18005`，API `http://192.168.50.123:8000` |
| NAS 部署路径 | `/volume3/docker/agno-demo-os`（宿主机 bind mount 到容器 `/app`） |

## 2. 技术栈与部署结构

### 2.1 前后端

- **后端**：Python 3.12 + FastAPI + Agno Agent 框架，入口 `app/main.py`。
- **前端**：单文件 `frontend/index.html` + Nginx 静态托管，使用本地 `tailwindcss.js` 与 `marked.min.js`，无外部 CDN。

### 2.2 数据层（重要）

- **项目代码 / Git / compose**：位于 NAS `/volume3/docker/agno-demo-os`。
- **demo-os-api 源码挂载**：`/volume3/docker/agno-demo-os` bind mount 到容器 `/app`，api 容器内代码变更随宿主机同步，无需 rebuild api 镜像。
- **物业 Demo 业务事实数据**：实际使用 SQLite，路径为 `/volume1/docker/volumes/agno-demo-os/property-data/property_demo.db`。工单、知识文档、Badcase、Skill、Agent、Trace 等物业运行数据均以该 SQLite 为准。
- **本地 RAG 索引**：位于 `/volume1/docker/volumes/agno-demo-os/property-data/rag_index`。
- **Postgres/pgvector**：`demo-os-db` 容器运行，数据卷 `/volume3/docker/volumes/agno-demo-os/pgdata`，端口 `5433:5432`。它是 AgentOS 相关依赖，**不是**物业业务知识库的事实数据源。

### 2.3 Docker 服务

服务名与角色：

- `demo-os-db`：Postgres/pgvector，数据卷 `/volume3/docker/volumes/agno-demo-os/pgdata`。
- `demo-os-api`：FastAPI 服务，源代码通过 `.:/app` bind mount 热重载；运行时命令 `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`。
- `demo-os-web`：Nginx 静态前端，端口 `18005:80`。**无源码 bind mount**，`frontend/index.html` 更新后必须 rebuild `demo-os-web` 镜像才能生效。

### 2.4 NAS 部署约束

- 仅通过 `docker compose` 管理容器，不直接使用 `docker run`。
- compose 文件位于 `/volume3/docker/agno-demo-os/compose.yaml`。
- 容器数据卷统一挂在 `/volume1/docker/volumes/agno-demo-os/` 与 `/volume3/docker/volumes/agno-demo-os/` 下。
- 生产回滚：在宿主机执行 `git checkout <tag>` 后 `docker compose up -d --build demo-os-api demo-os-web`，数据库容器通常无需重建。

## 3. 核心运行边界

### 3.1 Agent 体系

系统保留 5 个官方 Agent，任何新增/临时 Agent 会在启动时被 `_migrate_v1_3_3_agents` 清理并归并：

| agent_id | 类型 | 职责 |
|----------|------|------|
| `router` | orchestration | 意图分类：maintenance / billing / complaint / customer_service / other |
| `maintenance` | vertical | 维修报修、工单查询 |
| `billing` | vertical | 费用、缴费、账单咨询 |
| `complaint` | vertical | 投诉、邻里纠纷、责任争议 |
| `customer_service` | vertical | 一般咨询、小区规定、服务承诺 |

### 3.2 Skill 机制

- Skill 存储在 SQLite `skills` 表，通过 `trigger_condition`（逗号/顿号分隔关键词）做关键字与双字 Jaccard 匹配。
- Router 可以看到所有 enabled Skill 用于意图判断；Vertical Agent 只能看到绑定到该 Agent 的 Skill。
- Skill 可能声明 `model_id`，但业主面聊天会强制忽略，统一使用 `deepseek-v4-flash`。

### 3.3 RAG

- 知识库文档存储在 Postgres，由 `rag_indexer` 构建本地向量索引。
- 检索阶段返回 `citations`（doc_id / doc_title / chunk_index / chunk_id / supporting text / score）。
- 当检索零召回时，`app/chat.py` 会自动创建 `category=knowledge_gap` 的 Badcase（source=auto）。

### 3.4 MCP

官方保留 3 个 MCP Server：

| Server | 绑定 Agent | 当前能力 |
|--------|------------|----------|
| `weather-server` | `maintenance` | 天气查询（只读） |
| `calendar-server` | `customer_service` | 当前日期、预约时间建议（只读） |
| `workorder-server` | `maintenance` | 工单数量、待办列表等只读查询 |

当前所有 MCP 工具均为只读。聊天中发现缺少写操作（如指派工单、预约上门）时，系统只会如实告知当前无法完成，**不会**在聊天阶段自动创建能力缺口草稿。

正确路径是：人工点踩或手工创建 Badcase → 分类为 `mcp_capability` → 执行 Darwin 分析 → 生成 `capability_gap` 草稿 → 人工确认并记录为产品待办。

### 3.5 Trace 与成本

- 每次聊天产生一个 `chat_traces` 记录。
- `model_calls` 按 `stage` 记录调用：router、vertical_agent、ab_test_a、ab_test_b、darwin。
- 业主面聊天固定 `model_id=deepseek-v4-flash`、`thinking_enabled=true`、`model_selection_reason=owner-facing default`。
- Darwin 与 A/B 测试使用 `deepseek-v4-pro`。
- 成本为本地估算：若价格表未配置或 provider 未返回 token，显示「无法估算」，不伪造金额。

### 3.6 Badcase 闭环

生命周期状态：`pending → classified → fixing → verifying → closed/rejected`。

主要来源：

- `manual`：业主在聊天中点踩（`thumb_down`），`chat_feedback` 创建。
- `auto`：RAG 零召回自动创建（仅 knowledge_gap）。

运营动作：分类 → Darwin 分析 → 生成三类草稿（knowledge / skill_prompt / capability_gap）→ 人工确认发布 → 真实重测 → 验证关闭。

## 4. 模型策略

| 场景 | 模型 | 说明 |
|------|------|------|
| 业主聊天（Router + Vertical Agent） | `deepseek-v4-flash` | 强制使用，Skill 中声明的 model_id 被覆盖 |
| A/B 测试 | `deepseek-v4-flash` vs `deepseek-v4-pro` | 仅 `/api/models/ab-test` 与 `/api/model-configs/ab-test` 使用，body 只接受 `prompt` |
| Darwin 深度分析 | `deepseek-v4-pro` | 独立 trace，stage=darwin，不修改代码/不自动完成业务操作 |

## 5. 已有验收脚本

| 脚本 | 用途 | 运行方式 |
|------|------|----------|
| `scripts/test_v1_3_3_badcase_closure.py` | API 验收：会话管理、MCP 能力缺口、知识库闭环、复合 RAG+MCP 回归 | `BASE_URL=http://192.168.50.123:8000 python scripts/test_v1_3_3_badcase_closure.py` |
| `scripts/browser_acceptance_v1_3_3.py` | Playwright 三端 Tab + Badcase 详情 + 成本治理 + 硬刷新复测 | `BASE_URL=http://192.168.50.123:18005 python scripts/browser_acceptance_v1_3_3.py` |

已知测试数据：运行脚本会产生 `test-v133` / `evidence-v133` 用户的测试会话、测试 Badcase、DEMO_TEST 知识库文档。脚本末尾会尝试清理 DEMO_TEST 文档与 Badcase，但当前会话删除接口未实现，测试会话可能残留。

## 6. 回滚方式

1. 在 NAS 宿主机进入 `/volume3/docker/agno-demo-os`。
2. 备份 `.env`：`cp .env .env.bak.$(date +%Y%m%d%H%M%S)`。
3. `git fetch origin --tags`。
4. `git checkout v1.3.2`（或目标 Tag）。
5. `docker compose -f compose.yaml up -d --build demo-os-api demo-os-web`。
6. 验证容器内 `git rev-parse HEAD` 与目标 Tag SHA 一致。

## 7. 安全红线

- 禁止在任何文档、脚本、日志中记录密码、Token、API Key、SSH 凭证。
- 禁止 `rm -rf` 删除 `/volume1` 或 `/volume3` 数据文件。
- 禁止 force push、reset --hard、删除业务会话/工单/知识库/Badcase。
- 修改配置文件前必须先备份为 `.bak`。
