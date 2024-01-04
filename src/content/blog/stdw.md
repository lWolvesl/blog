---
title: 'Stable Diffusion XL'
pubDate: 2022-08-25
description: 'Stable Diffusion'
heroImage: 'https://i.wolves.top/picgo/202403050802925.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2024-02-01 BY WOLVES</p>

##### About 
Stable Diffusion is a latent text-to-image diffusion model. Thanks to a generous compute donation from Stability AI and support from LAION, we were able to train a Latent Diffusion Model on 512x512 images from a subset of the LAION-5B database. Similar to Google's Imagen, this model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM. See this section below and the model card.

> Stable Diffusion Web UI

##### 项目地址

[github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui.git)

##### 常用模型/提示词网站

- [civitai](https://civitai.com/)
- [huggingface](https://huggingface.co/)

> Stable Diffusion Web UI On Linux

1.Launch webui.sh

    - For ubuntu, Please use user instead of root
    - webui will install requirements by itself in directory venv , which includes with downloading basic model

2.If you find the red line: Cannot locate TCMalloc. Do you have tcmalloc or google-perftool installed on your system? (improves CPU memory usage)
    
    - For ubuntu, Please test `apt-get install libgoogle-perftools4 libtcmalloc-minimal4 -y`

3.Because in linux without public Ipv4, you may use the proxy to access webui
    
    - Please write args in webui-user.sh
    - --listen --enable-insecure-extension-access --no-gradio-queue --share 

4.There are some args else

    - To set user and password for protect, you can use `--gradio-auth user:passwd`
    - To promote perfermence, you can use `--xformers` for enable xformers` support
    - To select GPU, you can use `--device-id 0` to set, the number is gpu index, and you should know sd is a text-to-image by one gpu, not design for muti-gpus


> ERROR List

- [1] [StableDiffusion WebUI 软件升级与扩展兼容 1](https://juejin.cn/post/7204356282588282940)
- [2] [StableDiffusion WebUI 软件升级与扩展兼容 2](https://juejin.cn/post/7204356282588282940)

