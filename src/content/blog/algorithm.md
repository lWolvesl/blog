---
title: 'Algorithm'
pubDate: 2011-09-01
description: '算法笔记'
heroImage: 'https://i.wolves.top/picgo/202411192043919.png'
---

<p style="color: aquamarine;text-align: center">POST ON 2024-11-19 BY WOLVES</p>

## 目录
- [A* 算法](#001)
- [Dijkstra 算法](#dijkstra-算法)
- [Bellman-Ford 算法](#bellman-ford-算法)
- [Floyd-Warshall 算法](#floyd-warshall-算法)
- [Kruskal 算法](#kruskal-算法)
- [Prim 算法](#prim-算法)
- [动态规划](#动态规划)
- [贪心算法](#贪心算法)
- [分治算法](#分治算法)
- [回溯算法](#回溯算法)
- [深度优先搜索](#深度优先搜索)
- [广度优先搜索](#广度优先搜索)


> A* 算法
<div id="001"></div>

- Description
    - A* 算法是一种启发式搜索算法，用于在图或树中找到最短路径。它结合了Dijkstra算法的优点和贪心最佳优先搜索的优点。
    - 在A*算法中，每个节点都有一个估计的代价（通常是到目标节点的距离），称为启发式函数。启发式函数的选择对算法的性能有很大影响，分为当前代价和预估代价(常用欧拉距离)。
    - A*算法使用一个优先队列来存储待扩展的节点，优先队列中的节点按照启发式函数的值进行排序。
    - 在每次扩展节点时，A*算法会选择启发式函数值最小的节点进行扩展，直到找到目标节点或队列为空。
    - A*算法的关键在于启发式函数的设计。一个好的启发式函数可以大大减少搜索的节点数，提高算法的效率。

