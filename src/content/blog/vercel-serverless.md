---
title: 'Vercel serverless'
pubDate: 2023-09-30
description: '利用Vercel免费部署云函数，完全实现后台无服务器运行'
heroImage: 'https://i.wolves.top/picgo/202310302334470.png'
---
<p style="color: aquamarine;text-align: center">POST ON 2023-10-21 BY WOLVES</p>

![](https://i.wolves.top/picgo/202310302317746.png)

![|inline](https://i.wolves.top/picgo/202310302332638.png)

> FREE

## 免费的云服务

- [官网介绍与教程](https://vercel.com/docs/functions/serverless-functions)

Vercel对个人开发者是免费的，其对爱好者有如下的免费资源/月，其中云函数

![|inline](https://i.wolves.top/picgo/202310302215122.png)

Vercel支持以下的语言进行搭建服务,甚至可以混合语言运行

![|inline](https://i.wolves.top/picgo/202310302324527.png)

## 一个简单的示例

[GitHub/lWolvesl/vercel-function-helloworld](https://github.com/lWolvesl/vercel-function-helloworld.git)

与普通构建vercel相同，使用vercel导入项目，即可运行api，以下两个api为示例项目的测试
```
https://hellofunction1.vercel.wolves.top/api/hello?name=Wolves
https://hellofunction1.vercel.wolves.top/api/index
```


> 教程

## 基础知识
注：设置domain时使用`cname-china.vercel-dns.com`可以防止vercel的dns污染，提高访问速度

