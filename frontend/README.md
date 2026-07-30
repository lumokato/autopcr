# AutoPCR Web 构建覆盖层

生产镜像不再下载滞后的 AutoPCR_Web Release，而是在 Docker 构建阶段使用固定源码提交并应用本目录的覆盖文件。

- 上游源码：`Lanly109/AutoPCR_Web`
- 固定提交：`8e329362cfdbf30c72f987116ed3c488e969aaef`
- 包管理器：`pnpm 11.9.0`
- 覆盖内容：收藏状态迁移、栏目执行模式、手动工具不显示启用勾选框，以及相关导入导出兼容处理。

更新前端时先在 AutoPCR_Web 工作区应用同一组覆盖并运行 `pnpm run build`，再更新本目录文件和 Dockerfile 中的固定提交。
