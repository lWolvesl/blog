---
title: 'Embed'
pubDate: 2025-01-01
description: 'Embed'
heroImage: 'https://i.wolves.top/picgo/202403060835672.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2022-06-01 BY WOLVES</p>

## A temperature check service

> requirements

- stm32f103c8t6
- ESP8266
- DS18B20
- DS3231

> software

- stm32CubeMX
- openocd
    - [mac] brew install open-ocd
- [stlink](https://github.com/stlink-org/stlink)
    - [mac] brew install
    - st-info - a programmer and chip information tool
    - st-flash - a flash manipulation tool
    - st-trace - a logging tool to record information on execution
    - st-util - a GDB server (supported in Visual Studio Code / VSCodium via the Cortex-Debug plugin)
    - stlink-lib - a communication library
    - stlink-gui - a GUI-Interface [optional]

> configuration

- Clion
  - 1. create a new project
           [](https://i.wolves.top/picgo/202403062009498.png)

  - 2. select ioc file and open stm32mutemx to create your .h file
           [](https://i.wolves.top/picgo/202403062011130.png)

  select your stm32 chip and generate code
  [](https://i.wolves.top/picgo/202403062014558.png)

  注意勾选头文件生成
  [](https://i.wolves.top/picgo/202403062016537.png)

  - 3. 生成文件后，clion会自动弹出选择配置文件,此处先不选,创建自定义cfg并重新选择

```cfg
    # SPDX-License-Identifier: GPL-2.0-or-later
    # This is an ST NUCLEO F103RB board with a single STM32F103RBT6 chip.
    # http://www.st.com/web/catalog/tools/FM116/SC959/SS1532/LN1847/PF259875

    source [find interface/stlink.cfg]

    transport select hla_swd

    source [find target/stm32f1x.cfg]

    reset_config none separate
```