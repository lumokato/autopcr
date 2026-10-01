# AutoPCR Web 构建覆盖层

生产镜像不再下载滞后的 AutoPCR_Web Release，而是在 Docker 构建阶段使用固定源码提交并应用本目录的覆盖文件。

- 上游源码：`Lanly109/AutoPCR_Web`
- 固定提交：`66b69f8d563185a28b836d77fc426463da1affc8`（对应 1.9.0 tag）
- 包管理器：`pnpm 11.9.0`
- 覆盖内容：仅保留**栏目执行模式**（ExecutionMode）：
  - `interfaces/Module.ts`：`ModuleResponse.execution_mode?` 字段与 `ExecutionMode` 类型。
  - `components/Account/Area.tsx`：按后端下发的 `execution_mode`（缺失时按栏目 key 推断）计算并传给 Module。
  - `components/Account/Module.tsx`：`manual` 栏目隐藏启用勾选框，仅 `daily` 栏目显示配置同步按钮。
- `pnpm-workspace.yaml`：保留 1.9.0 原生的 `packages` 与 `allowBuilds` 配置。

收藏、导入导出、账号页缓存、并发保存和账号忙碌互斥等能力直接使用 1.9.0 上游实现，不再用旧版覆盖层回退。

更新前端时先在 AutoPCR_Web 工作区应用同一组覆盖并运行 `pnpm run build`，再更新本目录文件和 Dockerfile 中的固定提交。
