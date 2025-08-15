# Git3

## git diff

* git diff 显示的是**工作区**和**暂存区**的差异
* git diff --staged 比较的是**暂存区**和**本地分支**的差异
* git diff HEAD 比较的是**工作区**和**本地分支**的差异
* git diff main..feature 两点比较，比较的是两个分支的**绝对状态**的差异也就是**最后一次提交**的差异
* git diff main ... feature 三点比较，是把**feature分支**与两个分支的**分叉点**进行比较
* github 的pull request 默认是...比较，可以在网址上直接修改查看..比较
* git diff [commit id1] [commit id2] 比较两个commit的状态差异

## git stash

* git stash 把**暂存区**的文件都存储起来
* git stash list 可以查看所有的stash
* git stash -a 会把**工作区**的改动也存储进来
* git stash apply stash@{0} 取出stash但是不删除这个stash，注意在powershell中要使用强引用''
* git stash drop 可以删除某个stash
* git stash pop 取出并且删除stash，类似于Python中的栈
* git tag v1.0.0 在本地分支当前位置打上一个标签
* git tag v1.0.0 [commit id] 这就是在具体的某次提交打上tag
* git push --tag 把tag推送到远端
* git tag -d v1.0.0 删除本地tag
* git push origin --delete v1.0.0 像删除远端分支一样，删除远端tag
* git rebase -i [commit id] 交互式的rebase，可以用来修改commit的历史，例如可以把三次提交squash进一次提交里，当然这个操作也快是使用git reset --mixed 来完成。

`tag 跟分支，tag 跟commit，commit 跟分支都是可以进行对比的`