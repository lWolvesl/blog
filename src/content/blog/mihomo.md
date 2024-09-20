---
title: 'mihomo instead of clash'
pubDate: 2018-01-01
description: 'a proxy tool'
heroImage: 'https://i.wolves.top/picgo/202409121257063.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2023-12-31 BY WOLVES</p>

> clash is dead, mihomo stands

> clash 已死，mihomo 当立

- `mihomo` 原名 `clash meta`， 由于`clash`已经凉透了，所有因此而改名，目前当小猫的内核绝大多数都是`mihomo`。
- `Docker`一行完成`mihomo`内核搭建，满足服务器使用需求

```shell
docker run -d --name mihomo -p 7890:7890 -p 9090:9090 -v 目录/config:/root/.config/mihomo -v 目录/ui:/ui metacubex/mihomo:latest
```

## 1.获取config文件

- ~~从订阅转换 - [订阅网站](https://acl4ssr-sub.github.io/)/`https://acl4ssr-sub.github.io/`~~
- 从`clash-verge-rev`中获取

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

## 2.放入`geoip`文件
- `geoip`文件是一个全球`ip`数据库，通过这个配置，可以让代理工具知道什么网站应该走本地网络，什么网络应该走飞机，实现动态分流
- 下载文件-注意下载后重命名(有的时候自动下载会下载失败,因此提前下载)
  - [geoip.dat](https://i.wolves.top/picgo/202409121253507.dat)
  - [geosite.dat](https://i.wolves.top/picgo/202409121254143.dat)
  - [geoip.metadb](https://i.wolves.top/picgo/202409121254751.metadb)


## 3.控制
获取`ui`(可选)
```shell
git clone https://github.com/metacubex/metacubexd.git -b gh-pages ./ui
```

![](https://i.wolves.top/picgo/202401080022882.png)

![](https://i.wolves.top/picgo/202401080023841.png)
