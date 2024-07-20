---
title: '相关代理设置大全'
pubDate: 2026-01-01
description: 'Alpine Linux'
heroImage: 'https://i.wolves.top/picgo/202401142216861.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2022-02-05 BY WOLVES</p>

- [控制台终端代理](#%E6%8E%A7%E5%88%B6%E5%8F%B0%E7%BB%88%E7%AB%AF%E4%BB%A3%E7%90%86)
- [Git代理](#git%E4%BB%A3%E7%90%86)
- [Python相关代理](#python%E7%9B%B8%E5%85%B3%E4%BB%A3%E7%90%86)
    - [conda](#conda)
    - [pip/mim](#pipmim)

#### 控制台终端代理

```shell
# linux
export http_proxy=http://192.168.0.102:7890
export https_proxy=http://192.168.0.102:7890
```

> 也可以将这两行写入用户的`~/.bashrc`中，之后启动终端将自动配置，此处的代理具体`协议/ip/端口`应当自行设置

#### Git代理

```shell
# linux / windows / mac 通用
# 长期使用
git config --global https.proxy http://192.168.0.102:7890
git config --global http.proxy http://192.168.0.102:7890
# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

> 此处的代理具体`协议/ip/端口`应当自行设置

#### Python相关代理

> 若在控制台设置了终端代理，则`python`执行时会自动走代理

##### conda

```shell
# conda
vim ~/.condarc
# 在其中加入proxy如下
proxy_servers:
  http: http://192.168.0.16:7890
  https: http://192.168.0.16:7890
  
# 也可以设置清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
```

##### pip/mim

```shell
# pip 在具体命令后加入 --proxy=192.168.31.5:7890
pip install xxx --proxy=192.168.0.16:7890
# 或使用清华源
pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```shell
# openmim 在具体命令后加入 --proxy=192.168.31.5:7890
mim install xxx --proxy=192.168.0.16:7890
```

> 此处的代理具体`协议/ip/端口`应当自行设置

## docker
```shell
# vim 用户/.docker/config.json
{
  "proxies": {
    "default": {
      "httpProxy": "http://192.168.0.102:7890",
      "httpsProxy": "http://192.168.0.102:7890",
      "noProxy": ""
    }
  }
}
```

```shell
mkdir -p ~/.config/systemd/user/docker.service.d
vim ~/.config/systemd/user/docker.service.d/proxy.conf

[Service]
Environment="HTTP_PROXY=http://127.0.0.1:57890/"
Environment="HTTPS_PROXY=http://127.0.0.1:57890/"
Environment="NO_PROXY=localhost,127.0.0.1,.example.com"

systemctl --user daemon-reload
systemctl --user restart docker.servic
```