# Git2

## Git分支相关命令

* git branch [-a] 查看本地分支，-a选项意思是查看所有分支（包括远端分支）
* git checkout -b [feature] 基于本地分支创建分支
* git push --set-upstream origin feature2 将分支推送到远端，并且命名为feature2
* git switch 切换分支
* git branch -d [feature2] 删除分支（若分支还没有合并，则要使用-D删除）
* git push origin --delete feature2 删除远端分支
* git checkout [远端分支] 将远端分支检出到本地

* git merge 合并分支，需要切换到另一个分支，再merge当前分支，同样可以merge远端的分支

```python
# fastforward 快速前进，这样可以少产生一次merge的commit记录

## 例如基于main分支新建一个feature分支，feature分支提交了两次，但是main分支没有改变，这个时候只需要main分支向前移动两步即可，因为两个分支没有分叉。

# no fastforward 即使可以快速前进，我仍然想要产生一次merge的commit记录

## 可以使用git merge --no-ff [feature] 命令来实现
```

* git rebase 变基，在fastforward状态下无需强制推送，其他需要强制推送

## 比较分支

* git log feature..main 显示main分支上有的而feature分支上没有的**commit**
* git log feature..main 显示git log feature..main和git log main..feature两种结果
* git diff feature..main 也可以用来比较，但是比较的是差异
* git merge --squash feature2 ，squash merge只是放到了**暂存区**，没有进行**commit**，可以把**feature2分支有的的而本分支没有的**多次提交，当做一次merge进本分支
* git rebase 和git merge 都可能遇到冲突，解决冲突即可，命令行有提示
* git cherry-pick 选择想要某个分支的某几次commit合并进本分支



