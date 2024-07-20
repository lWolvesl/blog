---
title: 'Alpine Linux'
pubDate: 2024-01-01
description: 'Alpine Linux'
heroImage: 'https://i.wolves.top/picgo/202401072324448.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2022-08-29 BY WOLVES</p>

## Alpine
> 作为Docker官方推荐的linux构建基础镜像，alpine体积极小，并且拥有完整的linux内核以及操作，使用apk包管理工具

## 服务

```
apk add openrc
rc-update add docker boot
```


## 镜像源 - 上海交大

```shell
echo "http://mirrors.sjtug.sjtu.edu.cn/alpine/latest-stable/main" > /etc/apk/repositories
echo "http://mirrors.sjtug.sjtu.edu.cn/alpine/latest-stable/community" >> /etc/apk/repositories
```

## 设置时间

```shell
apk add tzdata --no-cache
cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
echo "Asia/Shanghai" > /etc/timezone
```

## 查看端口占用

```shell
netstat -atunlp
```

## CPU计数器

```shell
apk add util-linux
```

