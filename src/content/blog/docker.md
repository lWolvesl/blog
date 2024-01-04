---
title: 'docker rootless , gpu and other'
pubDate: 2022-12-30
description: 'docker'
heroImage: 'https://i.wolves.top/picgo/202403060834296.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2024-03-24 BY WOLVES</p>

# Docker Rootless

- [Installation](#installation)
- [GPU support](#gpu-support)
- [容器运行后，如何增加/修改/删除存储卷](#%E5%AE%B9%E5%99%A8%E8%BF%90%E8%A1%8C%E5%90%8E%E5%A6%82%E4%BD%95%E5%A2%9E%E5%8A%A0%E4%BF%AE%E6%94%B9%E5%88%A0%E9%99%A4%E5%AD%98%E5%82%A8%E5%8D%B7)

> Docker Rootless 模式允许用户在不使用 root 权限的情况下运行 Docker 守护进程（Docker daemon）。

> 这种模式提高了系统的安全性，因为它减少了潜在的攻击面。Docker Rootless 模式允许用户在不使用 root 权限的情况下运行 Docker 守护进程（Docker daemon）。这种模式提高了系统的安全性，因为它减少了潜在的攻击面。

### Installation

首先是 Docker，官方已经支持 Rootless 部署，文档在 https://docs.docker.com/engine/security/rootless/ ，使用上分为两步：

1.管理员用 root 权限配置好各项依赖

2.每个用户跑一次 setup 脚本，然后正常用 docker

<h4>1.在安装docker后，增加以下依赖</h4>

```shell
sudo apt install uidmap dbus-user-session fuse-overlayfs slirp4netns docker-ce-rootless-extras
```

<h4>2.安装完以后，每个用户自己执行一个初始化脚本</h4>

```shell
dockerd-rootless-setuptool.sh install
```

之后就可以正常使用 docker 了，从 docker info 可以看到目前是 rootless 模式：

```shell
docker info | grep Context
```

> 说明

- 在 Docker Rootless 模式下，daemon.json 文件的位置和使用方式与传统的 Docker 安装有所不同。由于 Docker Rootless 模式下 Docker 守护进程（Docker daemon）运行在非 root 用户的环境中，因此 daemon.json 文件的位置也会有所变化。

- 在 Docker Rootless 模式下，daemon.json 文件通常位于用户的 $HOME 目录下的 .config/docker 目录中。具体路径如下：

```shell
~/.config/docker/daemon.json
```

- 如果你需要配置 Docker Rootless 模式的守护进程，可以在这个路径下创建或编辑 daemon.json 文件。例如，你可以使用以下命令来创建或编辑该文件：

```shell
mkdir -p ~/.config/docker
nano ~/.config/docker/daemon.json
```

- 在这个文件中，你可以添加或修改 Docker 守护进程的配置，例如镜像加速器、日志驱动、存储驱动等。以下是一个示例 daemon.json 文件的内容：

```json
{
  "registry-mirrors": ["https://your.mirror.url"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

- 保存并关闭文件后，Docker Rootless 模式的守护进程会在启动时读取这个配置文件。请确保在修改配置文件后重启 Docker 守护进程以使更改生效。你可以使用以下命令来重启 Docker Rootless 模式的守护进程：

```shell
systemctl --user restart docker
```

- 或者，如果你使用的是 dockerd-rootless.sh 脚本来启动 Docker 守护进程，可以重新运行该脚本：

```shell
dockerd-rootless.sh
```

- 这样，Docker Rootless 模式的守护进程将会使用新的配置。

<br><br>

### GPU support

> 为dockerRootless增加GPU支持

在`/etc/nvidia-container-runtime/config.toml`的
`[nvidia-container-cli]`下添加

```toml
no-cgroups = true
```

即可为无根运行提供gpu支持

```shell
docker run -it --rm --gpus all ubuntu nvidia-smi
```

<br><br>

### 容器运行后，如何增加/修改/删除存储卷

> 首先，关闭容器进入容器目录，目录中一般有如下文件，我们需要修改的就是 config.v2.json 和 hostconfig.json 文件
![](https://i.wolves.top/picgo/202407031834556.png)

> 进入vim后，由于docker默认会将文件格式变为不易读，可以使用python工具进行解析为易读性较好的文件，esc 输入下列文字即可
```shell
:%!python -m json.tool
``` 
> 1.修改容器的"config.v2.json"配置文件

在"MountPoints"数组，最后添加宿主机目录"/opt/file"映射到容器的"/file"目录下，如下
```json
"/file": {
            "Source": "/opt/file",
            "Destination": "/file",
            "Driver": "",
            "Name": "",
            "Propagation": "rprivate",
            "RW": true,
            "Relabel": "ro",
            "SkipMountpointCreation": false,
            "Spec": {
                "Source": "/opt/file",
                "Target": "/file",
                "Type": "bind"
            },
            "Type": "bind"
}
```

> 2.修改容器的"hostconfig.json"配置文件

在"Binds"数组，最后添加目录映射配置，如下：
```json
"Binds": [
        "/opt/file:/file"
]
```

> 最后一步，一定要重启docker进程，否则直接启动容器，docker会把你的修改给还原

```shell
# rootless 进程
systemctl --user restart docker
```