---
title: 'Gitea'
pubDate: 2019-01-01
description: 'Gitea'
heroImage: 'https://i.wolves.top/picgo/202401072332887.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2022-10-30 BY WOLVES</p>

# GITEA

> 作为GOGs的社群支持版本，Gitea在大量开发者的努力下，已经具备了几乎和gitlab相当的能力，并且极其轻量化

- 搭建

## Docker

- dockerfile
```dockerfile
FROM python:3.10-alpine3.20

RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apk/repositories
RUN apk add tzdata iproute2 git bash bash-completion --no-cache
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo "Asia/Shanghai" > /etc/timezone

RUN mkdir /data
RUN adduser -D git
RUN echo 'git:git' | chpasswd
RUN chown -R git:git /data

RUN pip3 install nbconvert -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```shell
docker run -itd --name gitea -p 3000:3000 --privileged -v /volume2/docker/gitea:/data gitea:py3.10-a3.20 /data/app/docker-entrypoint
```

## 自定义构建

- [设置apk和时间](../alpine)

```shell
apk add git bash --no-cache
adduer git
su git -c '/data/app/gitea web --config /data/gitea/conf/app.ini'
```

## jupyter支持
- This guide will show you how to configure an external renderer to display Jupyter Notebooks. However this guide will also work for other types of files and even binary files! The sky is the limit.
- In order to display something more attractive to the user we need some HTML, luckily Jupyter has a module called nbconvert:

```shell
# Install our converter software of choice on the Gitea machine:
sudo apt install python3-pip
pip3 install nbconvert
```

- then Configuring Gitea to use the converter
```ini
[markup.jupyter]
ENABLED = true
FILE_EXTENSIONS = .ipynb
RENDER_COMMAND = "jupyter nbconvert --stdout --to html --template full "
IS_INPUT_FILE = true

[markup.sanitizer.jupyter0]
ELEMENT = div
ALLOW_ATTR = class
REGEXP =
```


## GITEA支持镜像上传

- 支持的软件包管理器[](https://docs.gitea.com/zh-cn/usage/packages/overview#支持的软件包管理器)

目前支持以下软件包管理器：

| Name                                                         | Language   | Package client            |
| ------------------------------------------------------------ | ---------- | ------------------------- |
| [Alpine](https://docs.gitea.com/zh-cn/usage/packages/alpine) | -          | `apk`                     |
| [Cargo](https://docs.gitea.com/zh-cn/usage/packages/cargo)   | Rust       | `cargo`                   |
| [Chef](https://docs.gitea.com/zh-cn/usage/packages/chef)     | -          | `knife`                   |
| [Composer](https://docs.gitea.com/zh-cn/usage/packages/composer) | PHP        | `composer`                |
| [Conan](https://docs.gitea.com/zh-cn/usage/packages/conan)   | C++        | `conan`                   |
| [Conda](https://docs.gitea.com/zh-cn/usage/packages/conda)   | -          | `conda`                   |
| [Container](https://docs.gitea.com/zh-cn/usage/packages/container) | -          | 任何符合OCI规范的客户端   |
| [CRAN](https://docs.gitea.com/zh-cn/usage/packages/cran)     | R          | -                         |
| [Debian](https://docs.gitea.com/zh-cn/usage/packages/debian) | -          | `apt`                     |
| [Generic](https://docs.gitea.com/zh-cn/usage/packages/generic) | -          | 任何HTTP客户端            |
| [Go](https://docs.gitea.com/zh-cn/usage/packages/go)         | Go         | `go`                      |
| [Helm](https://docs.gitea.com/zh-cn/usage/packages/helm)     | -          | 任何HTTP客户端, `cm-push` |
| [Maven](https://docs.gitea.com/zh-cn/usage/packages/maven)   | Java       | `mvn`, `gradle`           |
| [npm](https://docs.gitea.com/zh-cn/usage/packages/npm)       | JavaScript | `npm`, `yarn`, `pnpm`     |
| [NuGet](https://docs.gitea.com/zh-cn/usage/packages/nuget)   | .NET       | `nuget`                   |
| [Pub](https://docs.gitea.com/zh-cn/usage/packages/pub)       | Dart       | `dart`, `flutter`         |
| [PyPI](https://docs.gitea.com/zh-cn/usage/packages/pypi)     | Python     | `pip`, `twine`            |
| [RPM](https://docs.gitea.com/zh-cn/usage/packages/packages/rpm) | -          | `yum`, `dnf`, `zypper`    |
| [RubyGems](https://docs.gitea.com/zh-cn/usage/packages/rubygems) | Ruby       | `gem`, `Bundler`          |
| [Swift](https://docs.gitea.com/zh-cn/usage/packages/rubygems) | Swift      | `swift`                   |
| [Vagrant](https://docs.gitea.com/zh-cn/usage/packages/vagrant) | -          | `vagrant`                 |

- nginx转发时应使用fullchain

| 文件名        | 文件作用                                                     |
| ------------- | ------------------------------------------------------------ |
| cert.pem      | 服务端证书                                                   |
| chain.pem     | 浏览器需要的所有证书但不包括服务端证书，比如根证书和中间证书 |
| fullchain.pem | 包括了cert.pem和chain.pem的内容                              |
| privkey.pem   | 证书的私钥                                                   |