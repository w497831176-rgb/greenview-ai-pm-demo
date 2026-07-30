# ADR-0008｜结构化 LaneDecision 与阶段上下文隔离

- 状态：Accepted
- 日期：2026-07-30
- 适用：V1.8.2-S10-B.2-Fix1

## 决策

自然语言在选择业务 Agent 前必须先形成 `A_SAFETY_HANDOFF`、`B_PROPERTY_GOVERNED` 或 `C_ISOLATED_GENERAL` 三选一的 `LaneDecision`。候选 Agent 先按 Lane 过滤，随后才允许 Router 在该集合内选择；跨 Lane 建议被拒绝并留证。

Router 与业务 Agent 不再读取 Agno 自动历史。运行时只从 `chat_messages` 注入最近 5 轮成功可见对话，并为两个阶段使用独立 session。整体为内部控制 JSON 的业务输出在用户交付前阻断，不自动重试。

## 原因

失败 Trace `421abf858b9a4869` 证明共用阶段历史会把 Router JSON 带入业务回答；单纯在 Router 之后做关键词补救也无法从结构上证明 Agent、Skill、RAG、MCP/Tool 没有跨域越权。

## 被否决方案

- 为 T1/T2 句子增加关键词特例：不能覆盖反例，且继续积累不可解释分支。
- Router 继续查看全部 Agent、事后改写结果：越权候选已进入控制路径，证据不充分。
- 增加第二个大模型分类器或新编排平台：对个人 Demo 复杂度过高，并增加费用。

## 影响

- 新增严格 Pydantic `LaneDecision`、`CapabilityDecision`，复用现有 Evidence Ledger JSON，不迁移数据库。
- 安全 Lane 不调用普通模型或物业能力；B/C Agent 和能力集合互斥。
- 关闭自动历史后，用户可见多轮语义由显式装配保持。
- 配置没有变化，因此 RuntimeRelease 保持 v27。

## 回滚

普通 revert 实现提交并仅重建 API。回滚不改写历史 Trace、业务数据、RuntimeRelease 或既有 Snapshot。
