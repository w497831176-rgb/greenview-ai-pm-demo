# ADR-0006：业主聊天运行时单轨化

- 状态：Accepted
- 版本：V1.8.1-rc.2
- 日期：2026-07-23

## 决策

业主聊天只有 `RuntimeCoordinator` 可以执行 Router、Agent、Skill、RAG、MCP、
受控业务动作、Trace 与 Cost。`app/chat.py` 和
`app/runtime/legacy_chat.py` 只负责稳定的 HTTP/SSE、历史、反馈和人工接管协议。

删除进程内 `RUNTIME_ENGINE=v17|v18` 开关、`_stream_v17_response` 以及四个旧静态
业务 Agent 工厂。V1.7 业务执行器不再与 V1.8 共存抢权。

## 保留

- Agno 2.6.21、AgentOS、AgentFactory、WorkflowFactory；
- Router 单例与平台动态垂直 Agent；
- Skill、MCP、RAG、HITL、Trace、Cost；
- RuntimeRelease、Session Snapshot、Evidence/Cost Ledger；
- 当前确定性工单草稿、确认、写库与回执链；
- Badcase/Evaluation 和成本治理现有能力。

## 回滚

运行时回滚不再依赖同进程旧代码开关，而使用 `v1.8.1-rc.1` 或 `v1.8.0`
annotated tag 及其对应容器镜像。数据库、业务数据和持久化卷不回滚、不删除。

## 结果

同一请求不再存在 V1.7/V1.8 两套业务执行器争夺控制权的可能；动态能力仍由
Published RuntimeRelease 和下一新会话 Snapshot 决定。
