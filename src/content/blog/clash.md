---
title: 'CLASH'
pubDate: 2020-01-01
description: 'a proxy tool'
heroImage: 'https://i.wolves.top/picgo/202401080005682.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2022-12-31 BY WOLVES</p>

> 作为一款使用Go语言编写的强大的网络代理工具，其能力不必多言，本文将描述如何在linux上使用(docker)

## 1.获取config文件

- 从订阅转换 - [订阅网站](https://acl4ssr-sub.github.io/)/`https://acl4ssr-sub.github.io/`

> clash的订阅文件的编码格式无法在linux中编译，会出现无法识别的情况。

在vscode中打开文件，在首几行加入配置外部控制的端口

```ini
external-controller : '0.0.0.0:9090'
```

外部控制密码（可选）

```ini
# 在external-controller下一行
secret : '你的密码'
```

> 完成后将config.yaml 传入服务器指定目录

## 2.使用docker-compose

`镜像源 dreamacro/clash:latest`

```docker-compose
version: '3'

services:
  # Clash
  clash:
    image: dreamacro/clash:latest
    container_name: clash
    volumes:
      - ./config.yaml:/root/.config/clash/config.yaml
    ports:
      - "7890:7890/tcp"
      - "7890:7890/udp"
      - "9090:9090"
    restart: always

  clash-dashboard:
    image: centralx/clash-dashboard
    container_name: clash-dashboard
    ports:
      - "7880:80"
    restart: always
```

> 此处代码解释，启用了两个容器，并且将当前文件夹中的config.yaml与容器中的config.yaml绑定，设置了指定的端口转发

## 3.运行

```shell
docker-compose up --build -d
```

## 4.控制

![](https://i.wolves.top/picgo/202401080022882.png)

![](https://i.wolves.top/picgo/202401080023841.png)
