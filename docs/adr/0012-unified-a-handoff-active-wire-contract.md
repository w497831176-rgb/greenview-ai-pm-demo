# ADR-0012：统一 A Handoff 活跃线协议与旁路退役

- 状态：Accepted
- 日期：2026-08-08
- 决策范围：目标①主聊天链、A 路活跃协议、人工协同入口
- 替代关系：替代 ADR-0011 中关于 A 路普通/安全产品子类型以及
  `A_SAFETY_HANDOFF` 活跃线值的决定；历史记录继续只读保留。

## 背景

最终聊天合同要求每个业主新气泡先经过唯一 Router。A 只表达“本轮进入人工
协同”，不再让普通人工、安全人工、按钮直提或本地策略诊断形成并行运行分支。
历史 Trace 和 Handoff 行可能仍保存旧枚举，不能迁移或改写。

## 决策

1. 新 Router、Lane SSE、聊天消息和 Trace 统一写入 `A_HANDOFF`；A 的
   `selected_agent_id` 必须为空。
2. A 使用 Router 的自然语言理由直接创建一个统一 Handoff，并短路 Agent、
   Skill、RAG、MCP/Tool、Proposal 和 ActionGateway。
3. `POST /api/chat/handoff` 与 `POST /api/chat/handoff-policy` 固定返回 `410`，
   且不得产生状态变化。工作人员 claim/reply/wait/resolve/close 生命周期接口保留。
4. 旧值 `A_SAFETY_HANDOFF` 只允许在历史数据反序列化时映射读取；任何新输出
   都不得再次生成该值，也不改写历史行。
5. `capability_decision` 仅在 Router 已冻结 Agent（或 A 已确定短路）后记录实际
   装配结果。它不参与路由、改选、拒答或覆盖 Agent 回答。

## 结果与验证

- A 只有一条生产执行分支 `_stream_unified_handoff`。
- 所有带 Draft/Proposal 的后续气泡仍先经过一次 Router。
- 公开 `RuntimeCoordinator.stream` 的离线集成测试在 Provider 边界使用 fake，
  不替换 `_stream_selected_agent`，并验证完整工单状态机、幂等与失败 Receipt。
- 回滚使用承载本 ADR 的普通 Git 提交执行 `git revert`；历史数据无需迁移。
