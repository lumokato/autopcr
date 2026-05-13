# Agent Tasks

## 同步上游并强推

当用户要求同步上游时，在当前 Git 仓库执行：同步 `upstream/main` 到当前分支，尽量自动解决冲突，然后强制推送到 `origin`。

要求：
- 先运行 `git status --short --branch`、`git remote -v`，确认当前分支、`origin`、`upstream`。
- 不要丢弃本地改动；如有未提交改动，先理解用途，必要时提交一个临时 checkpoint 后再继续。
- 执行 `git fetch upstream`，再执行 `git rebase upstream/main`。
- 如果出现冲突，读取冲突文件，按代码语义尽量解决；解决后执行 `git add <file>` 和 `git rebase --continue`，循环直到完成。
- 只有在无法判断正确保留哪一边逻辑、测试/运行结果明显异常、认证失败或远程不存在时，才停止并说明需要人工判断。
- rebase 完成后检查状态；如项目有明确测试/检查命令，运行最小必要验证。
- 最后执行 `git push -f origin <当前分支>`。该仓库只有本人使用，允许强制推送。
