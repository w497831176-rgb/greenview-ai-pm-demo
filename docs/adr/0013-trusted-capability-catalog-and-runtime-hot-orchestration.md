# ADR-0013｜可信能力目录与运行时热编排

- 状态：Accepted
- 日期：2026-08-08
- 决策范围：目标②能力供应链、运行控制面、RuntimeRelease、会话快照与 MCP 边界
- 替代：ADR-0006 中面向生产控制面的 Git 导入、动态 Tool Discovery 与读写模板；
  Living Contract 2.3 中“平台实时新增任意 Agent/Skill/RAG/MCP”的产品定义

## 背景

旧管理面把“能力供应链”和“运行时控制”混在一起：操作者可以从 UI/API 新增、
导入或编辑能力实现，再把数据库中存在的对象直接编译进 RuntimeRelease。这样既
无法证明对象经过代码评审，也会迫使发布器依据名称、绑定关系或自然语言元数据
猜测能力域和 Tool effect。

面试演示真正需要证明的是两件分离的事：

1. 新能力如何经过代码评审、结构校验、迁移/注册、测试和部署成为可信供应；
2. 已入库可信能力如何在不改代码、不重启服务的前提下热启停、热绑定、发布和
   回滚，并保持会话 Snapshot 不漂移。

## 决策

### 1. 代码评审控制的可信目录

系统新增代码级 `Trusted Capability Manifest`。只有 Manifest 中显式列出的稳定
技术 ID 才是可信能力。每条记录至少声明：

```text
stable_id
capability_type: agent | skill | knowledge | mcp_server | mcp_tool | system_tool
catalog_version
reviewed_content_hash
reviewed_artifact_hash: Skill references / MCP executable source
domain_scope: property | isolated_general | control_plane
trust_status: trusted
source: code_reviewed_manifest
effect: read (MCP Tool only)
```

Manifest 固定受控入库时的实现内容 Hash；同一稳定 ID 的 Prompt、Skill 内容、RAG
正文、MCP 启动定义或 Tool Schema 被直接改库后，也必须视为 drift 并阻止目录使用
和新 Release 发布。Skill 的实际引用文件与 MCP 可执行源码/代码包另有受控资产 Hash；
同路径替换同样阻止发布。虚拟环境和依赖缓存不进入资产 Hash，部署 `.env` 正文永不读取，
只固定环境变量键名。数据库名称、当前绑定、是否被其他 Agent 使用，以及 Prompt
中的声明都不得推导 trust、domain_scope 或 effect。数据库中未列入 Manifest 的历史对象可以
继续只读保留，但不得显示为可信目录项、不得绑定、不得启用为生产能力，也不得
进入新 RuntimeRelease。

当前数据模型没有独立 system Tool 表，因此目录明确报告该能力类型为空；不得把
MCP Tool、代码函数或 Agent 绑定关系伪装成独立 system Tool。

### 2. 运行控制面只操作既有可信对象

允许的 Draft 动作为：

- 对既有可信 Agent、Skill、RAG、MCP Server 启用或停用；
- 把既有可信 Skill、RAG、只读 MCP 绑定到既有可信垂直 Agent或解绑；
- 查看 Diff 与影响；
- 校验并发布不可变 RuntimeRelease；
- 查看历史 Release并受控回滚 current 指针。

以下后端入口返回 `410 Gone`，仅隐藏前端不构成完成：

- Agent 创建、复制、导入、删除或实现编辑；
- Skill/RAG 创建、Git/Zip/文件上传、内容编辑、删除和实现级回滚；
- MCP Server 创建、Git 导入、地址/启动参数修改、Tool Discovery 或 Tool 定义/
  policy 编辑；
- Darwin/Badcase 草稿直接应用到 Agent、Skill、RAG、MCP 或 Tool 实现；
- 将未知对象标为 trusted，或绕过 Draft/RuntimeRelease 改变当前生产配置。

新增能力仍然可扩展，但必须走代码评审、Manifest 注册、向后兼容迁移、确定性
测试和部署。该供应链不是运行控制面 CRUD。

### 3. 发布编译 fail closed

新 Release 编译器只编译 Manifest 中存在且结构一致的对象。发布校验必须拒绝：

- 未知/未受控对象或绑定；
- 缺少稳定 ID、catalog version/content hash、trust/source 或 domain_scope；
- Agent 与 Skill/RAG/MCP 的结构化域不兼容；
- `property` 能力绑定到 `isolated_general` Agent，反向亦然；
- control-plane 能力绑定到垂直 Agent；
- 任何非 read MCP Tool、写 MCP policy 或未知 effect；
- 旧式隐式“全部 RAG”绑定。

编译器、ToolGateway 和 MCP 执行边界三层都只允许 read MCP。业务写入只允许显式
内部服务 Action，例如确认后的 `work_order.create`；ActionGateway 不接受
`mcp.*`。

### 4. 会话热编排语义

Draft 启停/绑定在发布前不改变 current RuntimeRelease。发布新不可变 Release 后，
无需改代码或重启服务；新 Session 固定新 Release Snapshot，已开始的 Session
保持原 `release_id/snapshot_hash`。回滚只移动 current 指针，历史 Release 和既有
Snapshot 不可变。

Router 始终只获得启用 Agent 的技术 ID、名称、描述和 scope。Capability Manifest
及绑定信息只在 Router 冻结 Agent 后用于 AgentFactory/Gateway 装配，不能进入路由
Prompt，也不能触发第二次选 Agent。

## 结果

- “热拔插”被准确限定为可信目录内对象的运行时热编排，不再暗示任意插件即时
  进入生产。
- 历史对象、Release、Trace、Receipt、工单和业务数据不迁移、不删除、不改写。
- v27 已有 Session 继续读取原不可变 Snapshot；运行隔离按稳定 ID Manifest 校验，
  不回查 Draft 绑定或按名称猜测。每轮运行时只校验 Router 已冻结的选定 Agent，
  无关旧 Agent 的配置错误不能迫使本轮失败或换 Agent；新 Release 发布仍全图校验。
- 新增能力的交付速度仍由通用 Manifest/接口/数据结构支持，不演变为业务题库式
  `if/else`。

## 回滚

代码回滚使用普通 `git revert`。Runtime 配置回滚使用已发布 Release 的受控 rollback
接口；两者都不得重写历史 Release 或既有 Session Snapshot。
