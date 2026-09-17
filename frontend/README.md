# AutoPCR Web 构建覆盖层

生产镜像不再下载滞后的 AutoPCR_Web Release，而是在 Docker 构建阶段使用固定源码提交并应用本目录的覆盖文件。

- 上游源码：`Lanly109/AutoPCR_Web`
- 固定提交：`a486934361212aabdabd5cd14d83f4fc1cb7a9c9`（对应 1.8.0 tag）
- 包管理器：`pnpm 11.9.0`
- 覆盖内容：仅保留**栏目执行模式**（ExecutionMode）：
  - `interfaces/Module.ts`：`ModuleResponse.execution_mode?` 字段与 `ExecutionMode` 类型。
  - `components/Account/Area.tsx`：按后端下发的 `execution_mode`（缺失时按栏目 key 推断）计算并传给 Module。
  - `components/Account/Module.tsx`：`manual` 栏目隐藏启用勾选框，仅 `daily` 栏目显示配置同步按钮。
- pnpm-workspace.yaml 覆盖：声明 `allowBuilds: esbuild`（1.8.0 已原生携带同内容，保留以固定构建行为）。

注：收藏（黄星/只显示收藏/导入导出收藏）、账号页"立刻清理"按钮与状态徽章已在 1.8.0 上游原生实现，与本仓库历史覆盖的 localStorage 格式（`autopcr_fav_<alias>`、`_fav_` 前缀）完全兼容，不再需要覆盖文件。

更新前端时先在 AutoPCR_Web 工作区应用同一组覆盖并运行 `pnpm run build`，再更新本目录文件和 Dockerfile 中的固定提交。
