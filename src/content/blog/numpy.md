---
title: 'Numpy'
pubDate: 2011-11-01
description: 'Numpy'
heroImage: 'https://i.wolves.top/picgo/202412281117580.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2024-12-28 BY WOLVES</p>

## 目录
- [sum](#sum) - 向量化

# sum

1.底层实现：np.sum 是用 C 语言实现的，利用了底层的循环和优化技术。这使得它比 Python 的原生循环快得多。(位于umath模块中)
2.批量操作：np.sum 可以在整个数组上同时进行操作，而不是逐个元素地进行。这种批量操作减少了 Python 层面的循环开销。
3.内存布局：NumPy 数组在内存中是连续存储的，这使得对数组的操作可以利用 CPU 的缓存，从而提高计算速度。
4.并行化：在某些情况下，NumPy 可以利用多线程或 SIMD 指令来进一步加速计算。

```python
import numpy as np

# 创建一个二维数组
array = np.array([[1, 2, 3], [4, 5, 6]])

# 对整个数组求和
total_sum = np.sum(array)

# 对每一列求和
column_sum = np.sum(array, axis=0)

# 对每一行求和
row_sum = np.sum(array, axis=1)

print("总和:", total_sum)
print("列和:", column_sum)
print("行和:", row_sum)
```

```html
<div>
    <p>总和: 21</p>
    <p>列和: [5 7 9]</p>
    <p>行和: [ 6 15]</p>
</div>
```

# column_stack

1.合并向量，但是每个单行向量会被视为列向量
2.np.column_stack 期望所有输入数组在第一个维度（行数）上具有相同的大小。
