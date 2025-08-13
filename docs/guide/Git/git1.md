# git-1

## git常用命令

* git --version 查看git版本号
* git clone 克隆仓库
* git status 查看当前git状态
* git add [文件名称] 把文件加入到**暂存区**
* git commit -m “评论” 把**暂存区**的文件加入到**本地仓库**
* git push 把**本地仓库**中的文件推送到**远端仓库**
* git restore --staged [test5.py] 后悔药的一种，把文件从**暂存区**取出到**工作区**
* git pull 更新**远端仓库**到**工作区**，建议多多使用`--rebase`参数，这样拉去更新的时候产生的是**线性**的提交记录
* git log 查看日志
* git rm 删除文件
* git mv 移动文件+重命名
* git reset --mixed(默认选项) [某次提交的sha256]，reset到**远端提交记录**之后必须使用强制推送才可以推送上去！
* git reset --hard 
* git reset --soft
* git push -f 强制推送

| 参数      | HEAD 移动 | 暂存区（Index） | 工作目录（Working Directory） | 风险等级 |
| :-------- | :-------- | :-------------- | :---------------------------- | :------- |
| `--soft`  | ✅         | 保留            | 保留                          | 低       |
| `--mixed` | ✅         | 重置            | 保留                          | 中       |
| `--hard`  | ✅         | 重置            | 重置（丢失数据！）            | 高       |

* git restore  -->（discard） 后悔药的一种，可以把工作区**修改或删除**的文件返回到工作区，不影响已经提交到**暂存区**的文件
* git show [commit id] 可以查看那次提交的详细信息
* git revert [commit id] 后悔药的一种，通过产生一次反向操作的提交来恢复到commit之前
* git commit --amend 后悔药的最后一种，可以修改最新一次提交，包括commit message和文件内容的修改，不产生新的commit记录

![](https://r2.tuple2.dpdns.org/tuple/pictures/20250813210746345.png)