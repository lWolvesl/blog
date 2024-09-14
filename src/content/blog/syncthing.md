---
title: 'syncthing'
pubDate: 2016-09-01
description: 'Open Source Continuous File Synchronization'
heroImage: 'https://i.wolves.top/picgo/202409141712459.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2024-09-14 BY WOLVES</p>

> 为了满足在两台服务器间实时同步数据，尝试了很多方案，但是都不尽人意，最后发现了这个Syncthing，他拥有自动同步，并且有GUI可以方便进行配置和管理。

## Usage
- docker-compose.yaml
```yaml
version: "3"
services:
  syncthing:
    image: syncthing/syncthing
    container_name: lb-project
    hostname: syncthing
    network_mode: bridge
    environment:
      - PUID=1001
      - PGID=1001
    volumes:
      - /wherever/st-sync:/var/syncthing
    ports:
      - 8384:8384 # Web UI
      - 22000:22000/tcp # TCP file transfers
      - 22000:22000/udp # QUIC file transfers
      - 21027:21027/udp # Receive local discovery broadcasts
    restart: unless-stopped
```

- 进入控制台 端口默认`8384`
![](https://i.wolves.top/picgo/202409141813378.png)

- 具体设置
  - 先在右下角增加设备，两个设备都添加对方后，会自动激活连接
  - 在文件夹中勾选共享后，服务器会自动同步，图三为正确设置后开始同步

![](https://i.wolves.top/picgo/202409141818567.png)
![](https://i.wolves.top/picgo/202409141820664.png)
![](https://i.wolves.top/picgo/202409141821445.png)