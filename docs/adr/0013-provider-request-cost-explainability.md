# ADR-0013：逐次 Provider 请求与成本解释投影

- 状态：Accepted
- 日期：2026-08-09
- 范围：现有 Trace 详情与成本区域的只读投影

## 决策

每个已持久化的物理 Provider 请求独立展示节点、请求模型、Provider 实际模型、Thinking、模型选择理由、Hit、Miss、Input、Output、其中 Reasoning、Total 和当次平台价格快照直接成本。缺少采集证据的字段显示“未采集”，不得从请求模型反推实际模型，也不得在前端反推 Token。

只读投影固定使用以下关系：

- `Input = Hit + Miss`
- `Output` 包含 `Reasoning`；Reasoning 仅作从属展示
- `Total = Input + Output`
- 成本仍取当次冻结价格快照：`Hit × Hit价 + Miss × Miss价 + Output × Output价`

RAG、Skill、MCP、Tool、ActionGateway 等非模型节点明确标记“未调用 Provider，无 Provider Token”。无法由结构化执行证据确认的历史节点显示“未采集”，不补造历史过程。

现有成本区域最多显示三条只读节约说明，每条标注“实测”“预期”或“未采集”，同时给出动作、成本结果和质量边界。没有真实测量时不得生成节省金额。

## 冻结边界

本决策不修改 ④-A 的历史 Token、北京时间归属、价格快照、成本公式、对账算法或任何历史记录。2026-08-01 冻结基准继续由既有账本和独立验收维护，本次只增加运行时只读解释字段。
