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

### 2.2 Gridient Descent

- 在当前的节点找下降速度最快的方向走一步，然后再找下一个方向，最终达到局部最小值`loacl minima`（贪心）

$$
w = w - \alpha\cdot\frac{\partial J(w,b)}{\partial w}
$$

$$
b = b - \alpha\cdot\frac{\partial J(w,b)}{\partial b}
$$

- $ \alpha $ is learning rate
- Simultaneously update
  - It mains we need calculate first , then update both of the value
- Here is negative gradient

**Learning Rate**

- Small
  - Gradient descent may be slow
- Large
  - Gradient descent may
    - Overshoot
    - Fail to converge
    - Diverge

![](https://i.wolves.top/picgo/202408041059271.png)

**Batch**

- Each step of gradient descent

### 2.3 Multiple features (variables)

$$
\ {\vec{x}^{(2)}_{3}} = \begin{pmatrix}x_1 & x_2 & x_3 & x_4\end{pmatrix}
$$

- $x_j = j^{th}$(feature)
- $n$ : number of features
- $\vec{x}^{(i)}$ : features of ($i^{th}$) training example
- $x_j^{(i)}$ : value of feature $j$ in ($i^{th}$) training example 

#### 2.3.1 Model

- Previously
  $$
  f_{w,b}(x) = wx + b
  $$

- Now

  multiple linear regression
  $$
  f_{w,b}(x) = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
  $$

  equals

  $$
  \vec{w} = \begin{bmatrix} w_1 & w_2 & w_3 & \cdots & w_n \end{bmatrix}
  $$

  $$
  \vec{x} = \begin{bmatrix} x_1 & x_2 & x_3 & \cdots & x_n \end{bmatrix}
  $$

  $$
  f_{\vec{w} \cdot,b}(\vec{x}) = \vec{w} \cdot \vec{x} + b
  $$

  It is not moltivariate regression(多个自变量和因变量的关系，即矩阵和矩阵)
