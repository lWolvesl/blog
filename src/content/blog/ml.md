---
title: 'Machine Learning for beginners'
pubDate: 2011-01-01
description: 'Machine Learning for beginners'
heroImage: 'https://i.wolves.top/picgo/202407100515101.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2024-03-05 BY WOLVES</p>

<link rel="stylesheet" href="/katex/katex.min.css">
<script defer src="/katex/katex.min.js"></script>
<script defer src="/katex/auto-render.min.js" onload="renderMathInElement(document.body);"></script>

# Machine Learning

## 1.First chapter - instruction

### 1.1 supervised learning

- regression
- classification

### 1.2 unsupervised learing

- Clustering 聚类算法
- Dimensionality reduction
    - 少所考虑的随机变量数量的过程，目的是获得一组主要变量。

### 1.3 Jupyter Notebook

## 2.Supervised Learing

### 2.1 Linear regression

$$
f(x) = wx+b
$$

- x is input feature / variable

A experiment:`https://github.com/mohadeseh-ghafoori/Coursera-Machine-Learning-Specialization.git`

- Cost function
    - Mean Squared Error, MSE - 均方差

$$
\text{Residual} = \hat{y}^{(i)} - y^{(i)}
\\
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (y_{w,b}(x^{(i)}) - y^{(i)})^2
$$

- goal

$$
\underset{w,b}{minimize}\ J(w,b)
$$

