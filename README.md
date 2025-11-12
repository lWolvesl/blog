## 文档顺序
- index             2000-01-01
- thinking          2000-06-01
- Daily             2000-09-01
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