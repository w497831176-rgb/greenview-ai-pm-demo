# ADR-0006｜产品化 MCP Git 导入与 Tool 分类

- 状态：Accepted
- 日期：2026-07-24
- 决策范围：目标 2 动态 MCP 新增、工具发现、读写分类与平台管理体验

## 背景

V1.8.1 RC5 的 MCP 管理页面直接要求操作者填写 `command`、`args` 和 `env`
JSON。保存后“查看工具”只读取缓存，并未调用后端已经存在的 `/discover`，
导致一个启动配置正确的新 MCP 仍显示空工具列表。

这只能证明开发者可以登记 stdio 启动参数，不能证明产品管理员可以从平台新增
一个插件并让它参与下一新会话，不符合目标 2 的“平台化插件能力”。

## 决策

1. MCP 管理的首选入口改为“从 Git 导入”，高级启动参数不再作为默认表单。
2. 公开 Git 仓库导入流程固定为：

   ```text
   clone
   → detect Python/Node
   → prepare isolated dependencies
   → detect stdio entrypoint
   → create MCP Draft
   → connect
   → discover/cache Tool schema
   → operator classifies read/create/update
   ```

3. 导入包保存在 `/app/data/mcp_packages`，由现有持久化业务数据卷承载，不写入
   Git 工作区；RuntimeRelease 仍只保存运行契约和版本证据。
4. Python 包使用隔离虚拟环境；Node 包由 API 镜像提供 Node/npm 并在包目录
   安装依赖。
5. Discovery 只说明 Tool 存在，不授予权限。Tool 必须由操作者选择：
   - 只读查询：`effect=read`；
   - 新增/写入：`effect=create`，运行时强制 Proposal；
   - 更新：`effect=update`，运行时强制 Proposal；
   - 暂不授权：`effect=unknown`，发布编译保持默认拒绝。
6. Agent 绑定、RuntimeRelease 发布、下一新会话和 Trace 证据边界保持不变。
7. 原 `command/args/env` 表单保留在“高级手动配置”，用于已有可执行入口和故障
   诊断；手动新增或编辑后也必须立即连接并刷新 Tool。

## 结果

- 普通平台操作者不再需要理解 stdio 命令和 JSON 参数才能完成首选导入流程；
- “导入成功”必须同时具备安装、连接和 Tool Schema 证据；
- 读写权限不会因 Git 仓库的自述或 Tool 名称自动生效；
- Git 导入仍属于 Draft，必须绑定 Agent、校验发布并在下一新会话真实验收；
- 不支持或构建失败的仓库必须显示真实失败阶段，不能创建假 Tool。

## 非目标

- 私有仓库 Token 托管；
- 任意语言和任意远程传输协议；
- 生产级插件沙箱、供应链扫描和资源隔离；
- 自动授权 destructive Tool。

上述项目不阻塞个人面试演示中的 Python/Node 公共 MCP Git 导入。
