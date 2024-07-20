---
title: 'OPENWRT'
pubDate: 2017-01-01
description: 'OPENWRT'
heroImage: 'https://i.wolves.top/picgo/202401072337181.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2023-12-09 BY WOLVES</p>


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