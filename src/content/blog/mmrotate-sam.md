---
title: '一次图像分割的尝试'
pubDate: 2022-09-29
description: 'mmrotate-sam'
heroImage: 'https://i.wolves.top/picgo/202401142147310.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2024-01-14 BY WOLVES</p>

> 基础库为[mmrotate](https://github.com/open-mmlab/mmrotate.git),而[mmrotate_sam](https://github.com/open-mmlab/playground/tree/main/mmrotate_sam)模块是在其基础上加入SAM和弱监督水平框检测，实现旋转框检测，从此告别注释旋转框的繁琐任务！

#### 1.首先根据[mmrotate](https://github.com/open-mmlab/mmrotate.git)的仓库指引，搭建mmrotate的环境

> 注意本教程全程按照给定shell命令挨个执行即可

> 若需要代理,请移步<a href="../allproxy" target="_blank">代理设置</a>

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
```
> 注意，此处的`mmcv-full`未指定版本，后续还需要安装`mmrotate_sam`的特殊版本依赖

#### 2.进入[mmrotate_sam](https://github.com/open-mmlab/playground/tree/main/mmrotate_sam)的仓库进行其余依赖安装

- 首先在`mmrotate`的跟目录下创建`mmrotate_sam`文件夹并将以下三个文件放入

![](https://i.wolves.top/picgo/202401142154222.png)

```shell
mkdir mmrotate_sam
cd mmrotate_sam
wget https://raw.githubusercontent.com/open-mmlab/playground/main/mmrotate_sam/data_builder.py
wget https://github.com/open-mmlab/playground/raw/main/mmrotate_sam/demo_zero-shot-oriented-detection.py
wget https://github.com/open-mmlab/playground/raw/main/mmrotate_sam/eval_zero-shot-oriented-detection_dota.py
git clone https://github.com/Li-Qingyun/sam-mmrotate.git
mv sam-mmrotate/configs ../mmrotate/configs
rm -rf sam-mmrotate
```

> 由于这个项目又是基于[sam-mmrotate](https://github.com/Li-Qingyun/sam-mmrotate.git)项目优化而来的，因此还需下载其文件

- 继续安装其需要的依赖

```shell
mim install mmengine 'mmcv>=2.0.0rc0' 'mmrotate>=1.0.0rc0'

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx

mkdir ../models
wget -P ../models https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget -P ../models https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth
```

#### 3.执行

```shell
python demo_zero-shot-oriented-detection.py \
    (所需分割的图片的路径) \
    ../mmrotate/configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py \
    ../models/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth \
    --sam-type "vit_b" --sam-weight ../models/sam_vit_b_01ec64.pth --out-path ./output.png
```

> 注意此处要传入所需分割图片的路径，当执行完成后，将在当前文件夹生成`output.png`为分割后图片，效果如下

![分割前](https://i.wolves.top/picgo/202401142209335.png)

![分割后](https://i.wolves.top/picgo/202401142209222.png)