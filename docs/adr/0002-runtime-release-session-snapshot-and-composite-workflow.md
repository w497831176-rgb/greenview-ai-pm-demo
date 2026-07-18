# ADR-0002｜显式发布、会话快照与组合验收

- 状态：Accepted
- 日期：2026-07-18
- 决策范围：V1.8 RuntimeRelease、动态能力生效、企业级同时命中验收

## 背景

平台配置若在每条消息中直接读取，会让同一会话中途改变 Agent、Skill、RAG
或 MCP 能力；若把只读咨询与写操作塞进同一无状态 Agent 调用，又会让模型
抢走提交权。另一方面，企业级验收仍需要证明多个技术栈可以在一次业务流程中
共同命中。

## 决策

1. 平台 CRUD 只产生 Draft；通过统一校验后发布不可变 `RuntimeRelease`。
2. 新会话第一次运行时生成 `RunConfigSnapshot`，后续运行不热切换 Release。
3. Skill 正文与 references、RAG Chunk 内容与 Hash、Agent/Tool/模型/价格绑定均
   进入 Release；凭据只保留环境变量名，不进入 Release。
4. 只读 MCP 可在发布白名单内自动执行；create/update MCP 只生成持久化
   Proposal，用户确认后由 ActionGateway 精确调用，真实资源 ID 是 committed
   Receipt 的必要条件；delete/unknown 默认拒绝。
5. 三条 Workflow 的授权边界保持分离。企业级“同时命中”由
   `composite_acceptance` 在一个 Agno Workflow run 中编排只读咨询和受控操作
   Step；组合不合并授权，也不取消 HITL 暂停。

## 结果

- 新增能力在发布后的下一新会话真实生效，旧会话可复现；
- 配置变化、历史 Citation 和成本证据不会被当前数据库内容静默改写；
- 组合演示可以同时展示 Router、Agent、Skill、RAG、读 MCP、HITL、写 MCP、
  Receipt、Trace 和 Cost，同时仍能解释每一步由谁授权；
- 回滚只移动当前 Release 指针，已有会话继续使用原 Snapshot。
