---
title: powershell 美化
date: 2019-6-30
tags: [Powershell, Scoop]
---
让powershell 变得好看一点

## 字体

在中文环境下，默认 PowerShell 的新宋体很丑。然而由于默认 PowerShell 终端是一个非常底层的应用，其界面甚至没有利用 WinForm、WPF 等通用 UI 渲染框架来实现，而是直接调用底层 Windows API 来实现，因此其字体要求非常严格。

目前据我所知，唯一支持这一严格要求的字体（中文环境下）只有「Sarasa Gothic / 更纱黑体 / 更紗黑體 / 更紗ゴシック」（后面简称更纱黑体），可以从 [更纱黑体的 GitHub](https://github.com/be5invis/Sarasa-Gothic/releases)页面下载。
<!-- more -->
## 配色

微软官方提供了一个更换 PowerShell 配色的小工具：ColorTool.exe，可以使用 Scoop安装：
```
scoop install colortool
```

ColorTool 使用非常简单。工具自带了几种主题，我们可以利用下面这个命令进行查看：
```
# 注：-s 代表 schemes
colortool -s
```
几个自带配色主题
其中前面列出的几个 .ini 和 .itermcolors 就是主题配置文件，我们可以直接通过下面这个命令设置主题：
```
# 临时查看
colortool <主题名称>
# 定义默认值
colortool -d <主题名称>
```
比如我们希望将主题配色更换为 OneHalfDark.itermcolors，只需要输入下面这个命令就可以更换并预览更新：
```
colortool OneHalfDark.itermcolors
```
*更换主题*
*由于 ColorTool 直接支持 iTerm 主题配置文件，因此我们可以在 iterm2colorschemes 这个网站找到我们想要的主题背景进行配置，方法和上面介绍的一样：在 PowerShell 中定位至你希望更换的主题文件，使用命令 colortool <主题名称>.itermcolors 进行配置即可。同时，如果你对上面的主题都不满意，你也可以直接在这个网站： terminal.sexy 自行配置自己想要的主题，并通过同样的方式进行应用。*
#### scoop 的安装
在 PowerShell 中输入下面内容，保证允许本地脚本的执行：
```
set-executionpolicy remotesigned -scope currentuser
```
然后执行下面的命令安装 Scoop：
```
iex (new-object net.webclient).downloadstring('https://get.scoop.sh')
```
静待脚本执行完成就可以了，安装成功后，让我们尝试一下：
```
scoop help
```

## 更好看

我们通过在 PowerShell 中执行下面的命令安装配置 oh-my-posh。

安装 posh-git 和 oh-my-posh 这两个模块
```
Install-Module posh-git -Scope CurrentUser 
Install-Module oh-my-posh -Scope CurrentUser
```
让 PowerShell 主题配置生效
新增（或修改）你的 PowerShell 配置文件
```
# 如果之前没有配置文件，就新建一个 PowerShell 配置文件
if (!(Test-Path -Path $PROFILE )) { New-Item -Type File -Path $PROFILE -Force }
```
```
用记事本打开配置文件
notepad $PROFILE
在其中添加下面的内容
Import-Module posh-git 
Import-Module oh-my-posh 
Set-Theme Sorin
```
其中最后一句 Set-Theme <主题名> 就是配置主题的命令,然后就是这样
![powershell](https://image-1252432001.cos.ap-chengdu.myqcloud.com/powershell/powershell.png)
