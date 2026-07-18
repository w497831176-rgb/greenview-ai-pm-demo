# 05｜部署、数据与操作边界

## 1. NAS 与容器

| 项目 | 当前约定 |
| --- | --- |
| NAS | Synology DS423+，局域网直接 SSH（端口由用户在会话中临时提供） |
| 代码/compose | `/volume3/docker/agno-demo-os` |
| API 容器 | `demo-os-api` |
| Web 容器 | `demo-os-web` |
| Postgres 容器 | `demo-os-db` |
| API | NAS `8000` |
| Web | NAS `18005` |
| 物业数据卷 | `/volume3/docker/volumes/agno-demo-os/property-data` → `/app/data` |
| 企业 Skill 只读挂载 | `/volume3/docker/volumes/agno-demo-os/skills` → `/app/enterprise/skills:ro` |

禁止再写回 `/volume1`。历史文档中若出现 `/volume1/docker/volumes/...`，一律视为旧信息，必须以当前 compose 和 Docker mount 为准。

## 2. 当前运行环境特征

- compose 的 API 使用源代码 bind mount 和 `uvicorn --reload`；这是演示开发环境，不是生产部署。
- 前端是 Nginx 静态镜像，改 `frontend/index.html` 后必须 rebuild/restart `demo-os-web` 才会生效。
- 仅改 Python bind mount 代码时，可以只重启 `demo-os-api`；但仍需做一次健康检查。
- 已存在 `.env` 修改、`.bak`、`data/`、`enterprise/` 等用户遗留文件。它们不是本任务的授权删除范围。

## 3. Git / NAS 操作规范

1. 新会话可使用用户临时在聊天中给出的 NAS SSH 凭据和 Git token；不得把它们写入仓库、文档、Git remote URL、Git config 或最终报告。
2. 先只读检查 `git status --short`。若工作区 dirty，列出文件并停止，除非用户明确允许某个安全做法。
3. 不使用 `git reset --hard`、`git clean`、force push、删除数据卷。
4. 需要发布时：精确 stage 本次文件 → 普通 commit → 普通 push → pull/fast-forward → 重启受影响服务 → 最小健康检查。
5. 不跑 Playwright、浏览器容器、全量外网下载或长时间模型压测；用户自行在真实前端点测。

## 4. 成功发布的最低证据

- GitHub `main` 与 NAS 运行 commit 一致；
- 受影响 API 有 200 健康响应；
- 若有真实写操作，至少有一条低成本定向后端验收并清理自己的 `DEMO_TEST_*` 数据；
- 不把“静态检查通过”写成“端到端验收通过”；
- 用户手点前端所得的 Bug 应优先于模型自报结果。

## 5. 不要泄漏到交接资料中的内容

- NAS 密码、Git PAT、DeepSeek/Kimi API Key、`.env` 全文；
- 真实用户手机号、住址、私人数据；
- 临时脚本中出现过的凭据。
