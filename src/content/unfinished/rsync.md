---
title: 'Rsync双向备份'
pubDate: 2023-08-10
description: 'rsync + inotify'
heroImage: 'https://i.wolves.top/picgo/202401082231827.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2024-01-08 BY WOLVES</p>

> 服务器文件经常需要双向同步，rsync能够同步，但是不能实时监控文件变化，而inotify可以监听文件的增删改,为了方便操作，专门配置了docker镜像
> 