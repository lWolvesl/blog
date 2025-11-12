---
title: 'Daily'
pubDate: 2000-09-01
description: 'Daily'
heroImage: 'https://i.wolves.top/picgo/202511120000.png'
---

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

> 2025/11/12

[质数分解] - [最小公倍数] - [最大公约数]
任何一个数都可以表示为质数幂的乘积

eg. <p> $12 = 2^2 \times 3^1$ </p>
    <p> $8 = 2^3 \times 3^0$ </p>

GCD 被转换为对幂求最小值：<p> $gcd(12,8) = 2^{min(2,3)} \times 3^{min(1,0)} = 2^2 \times 3^0 = 4 $ </p>
LCM 被转换为对幂求最大值：<p> $lcm(12,8) = 2^{max(2,3)} \times 3^{max(1,0)} = 2^3 \times 3^1 = 24 $ </p>
