# Astro Starter Kit: Blog

```sh
npm create astro@latest -- --template blog
```

[![Open in StackBlitz](https://developer.stackblitz.com/img/open_in_stackblitz.svg)](https://stackblitz.com/github/withastro/astro/tree/latest/examples/blog)
[![Open with CodeSandbox](https://assets.codesandbox.io/github/button-edit-lime.svg)](https://codesandbox.io/p/sandbox/github/withastro/astro/tree/latest/examples/blog)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/withastro/astro?devcontainer_path=.devcontainer/blog/devcontainer.json)

> 🧑‍🚀 **Seasoned astronaut?** Delete this file. Have fun!

![blog](https://github.com/withastro/astro/assets/2244813/ff10799f-a816-4703-b967-c78997e8323d)

Features:

- ✅ Minimal styling (make it your own!)
- ✅ 100/100 Lighthouse performance
- ✅ SEO-friendly with canonical URLs and OpenGraph data
- ✅ Sitemap support
- ✅ RSS Feed support
- ✅ Markdown & MDX support

## 🚀 Project Structure

Inside of your Astro project, you'll see the following folders and files:

```text
├── public/
├── src/
│   ├── components/
│   ├── content/
│   ├── layouts/
│   └── pages/
├── astro.config.mjs
├── README.md
├── package.json
└── tsconfig.json
```

Astro looks for `.astro` or `.md` files in the `src/pages/` directory. Each page is exposed as a route based on its file name.

There's nothing special about `src/components/`, but that's where we like to put any Astro/React/Vue/Svelte/Preact components.

The `src/content/` directory contains "collections" of related Markdown and MDX documents. Use `getCollection()` to retrieve posts from `src/content/blog/`, and type-check your frontmatter using an optional schema. See [Astro's Content Collections docs](https://docs.astro.build/en/guides/content-collections/) to learn more.

Any static assets, like images, can be placed in the `public/` directory.

## 🧞 Commands

All commands are run from the root of the project, from a terminal:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Installs dependencies                            |
| `npm run dev`             | Starts local dev server at `localhost:4321`      |
| `npm run build`           | Build your production site to `./dist/`          |
| `npm run preview`         | Preview your build locally, before deploying     |
| `npm run astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `npm run astro -- --help` | Get help using the Astro CLI                     |

## 👀 Want to learn more?

Check out [our documentation](https://docs.astro.build) or jump into our [Discord server](https://astro.build/chat).

## Credit

This theme is based off of the lovely [Bear Blog](https://github.com/HermanMartinus/bearblog/).

## 文档顺序
- index             2000-01-01
- thinking          2000-06-01
- dl                2010-01-01
- ml                2011-01-01
- algorithm         2011-09-01
- numpy             2011-11-01
- amx               2012-01-01
- stdw              2013-01-01
- sampaper          2014-01-01
- mmrotate-sam      2015-01-01
- openwrt           2016-01-01
- MIAX1800          2016-06-01
- syncthing         2016-09-01
- docker            2017-01-01
- wireguard         2017-06-01
- how to read paper 2017-09-01
- mihomo            2018-01-01
- gitea             2019-01-01
- Apple Grapher     2020-01-01
- Vercel serverless 2022-01-01
- stools            2023-01-01
- Alpine Linux      2024-01-01
- embed             2025-01-01
- allproxy          2026-01-01
- CUT               2999-12-31


## 增加latex支持

- 下载katex文件到本地
```shell
# font 此处仍需移动文件位置到public/katex/fonts下
wget -r -np -nH --cut-dirs=2 -R "index.html*" -P public/katex/fonts https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/fonts/
# 下载css文件
wget https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css
wget https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js
wget https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js
```

1.在全文引用`script`
```html
<link rel="stylesheet" href="/katex/katex.min.css">
<script defer src="/katex/katex.min.js"></script>
<script defer src="/katex/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
<!-- Auto-render inline math formulas -->
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false}
            ]
        });
    });
</script>
```

2.在使用latex脚本的位置使用(可无需使用p标签/矩阵需要使用标签包含)
```html
<p>
    考虑一个封闭曲面 \( S \)，其第二型曲面积分可以表示为：
    \[
    \oiint_S \mathbf{F} \cdot d\mathbf{S}
    \]
    其中，\(\mathbf{F}\) 是一个向量场，\(d\mathbf{S}\) 是曲面 \(S\) 的微分面积元素。
</p>
<p>
    具体地，如果 \(\mathbf{F} = P\mathbf{i} + Q\mathbf{j} + R\mathbf{k}\)，则积分可以展开为：
    \[
    \oiint_S \mathbf{F} \cdot d\mathbf{S} = \oiint_S (P \, dy \, dz + Q \, dz \, dx + R \, dx \, dy)
    \]
</p>
矩阵
<p>
    $$
    \begin{pmatrix}
    a & b & c \\
    d & e & f \\
    g & h & i
    \end{pmatrix}
    $$
</p>
```