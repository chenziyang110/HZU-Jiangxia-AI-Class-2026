# AI课程学习笔记
姓名：杨忠赐
学号：2305070135

# PR提交：
1Fork他人的GitHub仓库,来创建一个完全独立的副本仓库
    通过GitHub仓库网页链接，进入他人GitHub仓库网页，点击"Fork"按钮
    【最终得到一个副本仓库：https://github.com/[你的用户名]/[原仓库名]
2Clone他人的GitHub仓库到本地，来得到一个本地文件夹（里面内容与GitHub仓库一样）
    进入“Git Bash here”操作界面，
    输入：git clone [Fork来的副本仓库的GitHub仓库网页链接]
3在本地文件夹里增删改文件【工作区的修改】
4提交“增删改”到本地仓库【放入暂存区、永久保存到本地仓库】
    进入“GBh”操作界面，
    输入：
         git add .
         git commit -m "[本次pr的解释说明]"
5推送到GitHub【推送到远程仓库】
    进入“GBh”操作界面，
    输入：
         git push origin main
6创建Pull Request（PR）
    浏览器访问副本仓库网页链接：https://github.com/[你的用户名]/[原仓库名].git
    点击 Pull requests → New pull request
    设置：
        base: chenziyang110/main (老师仓库)
        compare: yzc2453/main (你的仓库)
    填写标题：*******
    点击 Create pull request


# 更新PR提交：
$预先准备
1【创建分支、切换分支】
    # 在本地仓库（main分支）中：
    git branch 分支名称    # 创建分支
    git checkout 分支名称  # 切换分支
    # 或简写：
    git checkout -b your-branch-name


2同步上游仓库【"同步"Fork，获取老师仓库的最新修改】
第一步：添加上游仓库【首次更新运行即可，下次更新不用第一步】
    cd ~/Desktop/HZU-Jiangxia-AI-Class-2026

    # 查看当前远程仓库
    git remote -v
    # 应该只有 origin（你的GitHub）

    # 添加上游仓库（老师的）
    git remote add upstream https://github.com/chenziyang110/HZU-Jiangxia-AI-Class-2026.git

    # 再次查看
    git remote -v
第二步：拉取最新更新
    # 1. 切换到main分支
    git checkout main

    # 2. 从上游仓库拉取更新
    git fetch upstream

    # 3. 合并到你的本地main分支
    git merge upstream/main

    # 4. 推送到你的GitHub（更新你的Fork）
    git push origin main
第三步：在你的分支上同步
    # 切换到目的分支
    git checkout  目的分支名称

    # 合并main分支的最新内容（包含上游更新）
    git merge main

    # 解决可能出现的冲突（如果有）
    # git mergetool 或手动编辑文件

    # 推送到你的GitHub
    git push origin  目的分支名称


$分支提交pr
1创建分支
    # 进入项目
    cd HZU-Jiangxia-AI-Class-2026

    # 创建新分支（分支名要有意义）
    git checkout -b add-notes-20250121

    # 或更详细的分支名
    git checkout -b yangzhongci/update-notes
2在分支上增删改文件
3提交到分支
    # 返回项目根目录
    cd ../../..【cd 到HZU-Jiangxia-AI-Class-2026】

    # 提交
    git add .
    git commit -m "[本次pr的解释说明]"

    # 推送到你的GitHub（注意：推送到分支，不是main！）
    git push origin 目的分支名称



