---
title: 'wireguard'
pubDate: 2017-06-01
description: 'Fast VPN'
heroImage: 'https://i.wolves.top/picgo/202409201106809.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2024-09-20 BY WOLVES</p>

> WireGuard® is an extremely simple yet fast and modern VPN that utilizes state-of-the-art cryptography.

- `WireGuard` 在 `Linux` 内核 `5.4` 版本中被正式集成。此版本于 `2019 年 11 月`发布，之后的所有版本都原生支持 `WireGuard`。
- 在此之前，用户需要通过`外部模块或补丁`来使用 `WireGuard`。自 `5.4` 版本起，`WireGuard` 成为 `Linux 内核`的一部分，提供了更好的性能和安全性。
  - 因此，群晖(`7.2`的目前内核版本是`4.4.302`)需要先在第三方社群套件中安装`wireguard`才能继续安装，而内核大于`5.4`的任意`linux`发行版可以直接使用本`blog`中的`docker compose`快速搭建管理界面

- 此处使用了[wg-easy/wg-easy](https://github.com/wg-easy/wg-easy)项目，内部集成了`ui`界面方便管理，详细配置信息可以访问github访问
```yaml
version:'3.8'
services:
  wg-easy:
    environment:
      - LANG=chs
      - WG_HOST=1.wolves.top
      - PASSWORD_HASH=''
      - PORT=51880
      - WG_PORT=51820
      - WG_DEFAULT_ADDRESS=10.101.0.x
      - WG_DEFAULT_DNS=1.1.1.1
      - WG_MTU=1420
      - WG_ALLOWED_IPS=10.101.0.0/16,192.168.31.0/24
      - WG_PERSISTENT_KEEPALIVE=30
      - UI_TRAFFIC_STATS=true
      - UI_CHART_TYPE=2
    network_mode: "bridge"
    image: ghcr.io/wg-easy/wg-easy
    container_name: wireguard
    volumes:
      - ./config:/etc/wireguard
    ports:
      - "51820:51820/udp"
      - "51880:51880/tcp"
    restart: unless-stopped
    cap_add:
      - NET_ADMIN
      - SYS_MODULE
    sysctls:
      - net.ipv4.ip_forward=1
      - net.ipv4.conf.all.src_valid_mark=1
```

![](https://i.wolves.top/picgo/202409201104068.png)