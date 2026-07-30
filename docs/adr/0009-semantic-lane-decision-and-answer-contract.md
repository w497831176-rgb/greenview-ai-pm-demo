# ADR-0009｜结构化语义分流与 AnswerContract

- 状态：Superseded by ADR-0010
- 日期：2026-07-30
- 适用：V1.8.2-S10-B.2-Fix2
- 替代：ADR-0008 中基于确定性关键词 Lane 与 Lane 内 fallback 的决策方式

## 决策

每条新的自然语言请求先由当前 Published Snapshot 下的 Flash Router 做一次完整语义判断，输出严格 Schema 校验的 `LaneDecision`。Lane 只允许 `A_SAFETY_HANDOFF`、`B_PROPERTY_GOVERNED`、`C_ISOLATED_GENERAL` 和 `CLARIFY`；Schema 无效、目标 Agent 越过 Lane 边界或 Provider 失败时，本轮受控失败，不进行关键词 fallback，也不默认进入物业 Lane。

运行时再从 `LaneDecision` 确定性生成 `AnswerContract`，以结构化方式约束回答模式、Evidence 要求、能力权限、写入确认和 Handoff。最终交付门禁只检查 AnswerContract 与 Citation、成功 Tool 结果、Receipt 等结构化证据，不从模型自然语言中猜测事实类型。

`lane`、`request_kind` 与 `response_mode` 是正交产品字段：Lane 决定领域和权限边界，request_kind 描述用户任务类型，response_mode 描述最终执行方式。因此 C Lane 合法支持 `fact` 与 `general`，两者都生成 `safe_general`；`fact` 本身不决定 Evidence，B + fact 才因物业受控事实生成 `grounded_answer` 并要求合法 Evidence。

批准映射：A + emergency → emergency_handoff；B + fact/realtime_read/state_change/unsafe_request → grounded_answer/realtime_read/controlled_write/safe_refusal；C + fact/general/unsafe_request → safe_general/safe_general/safe_refusal；CLARIFY + ambiguous → clarify_only。其他跨领域或执行矛盾组合继续由 Schema 拒绝。

## 原因

Fix1 的影子评估证明关键词、正则和默认 B 无法理解对象、地点、否定、多意图与上下文；同时 Lane 正确并不能阻止无 Evidence 回答编造物业办理流程。语义决策和回答合同必须成为两个独立、可持久化、可验证的控制对象。

## 约束

- 测试集不得进入生产运行时，测试句子不得转成词表、正则、同义词或 case 分支。
- A 与 CLARIFY 不选择业务 Agent；B 只选 `property`，C 只选 `isolated_general`。
- C 跳过全部物业 Skill、RAG、MCP/Tool、ActionGateway；B 的事实回答必须由合法 Evidence 支撑。
- 不得为了 Schema 通过而把非物业知识事实强制改写为 `general`；C + fact 仍不得冒充实时、官方或物业依据。
- Router JSON、Prompt 与内部控制字段不进入业务 Agent 历史或用户回答。
- 保留现有多轮 Provider 请求逐次落账、Snapshot、ActionGateway、Citation 与成本合同。

## 影响

- 复用现有 Trace、`model_calls` 与 RunEvidenceLedger JSON，不新增数据库表或字段。
- 代码发布不改变 Published 能力配置，RuntimeRelease 保持 v27。
- 旧 Trace 继续按历史合同解释，不回填或改写。

## 回滚

普通 `git revert` Fix2.2 实现提交后，仅重建受影响的 API；不改写历史 Trace、业务数据、RuntimeRelease 或既有 Snapshot。也可恢复修改前备份 `/volume3/docker/agno-demo-os-backups/s10b2-fix2.2-pre-66a2ad9-20260730.tar.gz` 后重新构建 API。
