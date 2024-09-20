---
title: 'NEWWORK-MyHome'
pubDate: 2016-01-01
description: '家庭网络结构'
heroImage: 'https://i.wolves.top/picgo/202401072337181.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2024-06-09 BY WOLVES</p>

> 一图看完我的家用网络结构

![](https://i.wolves.top/picgo/202407301220389.png)

## IPV6

- 国内ISP逐步完全提供IPV6，虽然为动态，但是不影响使用
- 当前河南联通家庭网络最高可以申请/60的PD

### 设置IPV6

- 首先无论通过拨号还是`DHCP`获取到`IPV6`的设备，都需要在DHCP中设置不能禁止解析`IPV6`，IPv6长度默认64

![截屏2024-07-30 11.45.42](https://i.wolves.top/picgo/202407301145112.png)

- 针对通过`PPPOE`拨号获得IPV6的设备

> pppoe-wan 常规取消勾选忽略此接口，让wan接口成为主接口从而可以讲PD(前缀委派Prefix delegation)下发,高级委托IPv6前缀勾选

![截屏2024-07-30 11.42.21](https://i.wolves.top/picgo/202407301142523.png)

> Lan 口常规不忽略此接口，设置为服务器模式，RA设置管制+其他

![截屏2024-07-30 11.46.32](https://i.wolves.top/picgo/202407301146348.png)

![截屏2024-07-30 11.46.59](https://i.wolves.top/picgo/202407301147133.png)

> Wan6 新建DHCPv6接口，防火墙和WAN同一个，DHCP忽略，IPv6全中继

![截屏2024-07-30 11.48.31](https://i.wolves.top/picgo/202407301148868.png)

![截屏2024-07-30 11.50.02](https://i.wolves.top/picgo/202407301150975.png)

- 针对通过`DHCP`获取到的`IPV6`(二级路由)

> lan，wan 口不忽略，IPV6全中继

![截屏2024-07-30 11.51.28](https://i.wolves.top/picgo/202407301151414.png)

> wan6 新建DHCPv6接口，防火墙同WAN，忽略接口三中继，高级双勾选

![截屏2024-07-30 11.52.40](https://i.wolves.top/picgo/202407301152695.png)

### 针对openwrt为拨号设备和DHCP设备

#### 内网中的IPV6设备无法被外网访问

- 首先明确一点，在接口处通过`PPPOE`已经获取到了公网地址，并且已经正常向下分发`IPV6`

![](https://i.wolves.top/picgo/202407301106099.png)

![](https://i.wolves.top/picgo/202407301107774.png)

- 如图，我当前的`PPPOE`已经获取到了`ISP(联通)`分发的`2408`前缀，内部网络向外访问正常，但是外部向内部访问具体网站无效，`ICMP`正常，内网通过`IPV6`访问内网服务正常，要解决这个情况，现在先解释几个问题：
  - 1、监听 - 自我部署服务大多数情况下默认监听IPV4，如`IPV4:8080`，此时是无法被任意域通过`IPV6`访问的，可以通过`nginx`代理监听或者设置访问监听如`::8080`
  - 2、安全性，正常情况下，所有的路由器默认禁止外部通过`IPV6`直接访问内部的访问，只给`ICMP`质询内网主机的权限，如下图防火墙设置，这种情况下保证了虽然主机在公网，但是也不会被攻击，是安全性考量。

![](https://i.wolves.top/picgo/202407301114527.png)

>  解决1: 设置服务监听所有接口



![截屏2024-07-30 11.22.31](https://i.wolves.top/picgo/202407301122657.png)

> 解决2: 设置wan域可以访问lan网域

![](https://i.wolves.top/picgo/202407301120716.png)




### OPENWRT

you can get anything at [official](https://archive.openwrt.org/releases/)

> fireware-bulid

[immortalwrt](https://firmware-selector.immortalwrt.org/)

![](https://i.wolves.top/picgo/202407201210249.png)

[openwrt.ai](https://openwrt.ai/)

![](https://i.wolves.top/picgo/202407201209579.png)

# OrayX3A

## Openwrt install

`https://www.emperinter.info/2023/10/01/how-to-flash-openwrt-to-oray-x3a/`

- enable ssh
    - Login in and use `http://10.168.1.1/cgi-bin/oraybox?_api=ssh_set&enabled=1` to enable
    - Use rsa to ssh `ssh -o HostKeyAlgorithms=ssh-rsa root@10.168.1.1`,default password is `oray@12#$%^78`
- please backup the official firmware
    - `dd if=/dev/mtd3 of=/tmp/firmware.bin`
    - `scp -o HostKeyAlgorithms=ssh-rsa -O root@10.168.1.1:/tmp/firmware.bin ./`
- Download your firmware
- Then mtd it `mtd write xxx.bin firmware`