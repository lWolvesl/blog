---
title: 'THINKING'
pubDate: 2011-06-01
description: 'some thinking'
heroImage: 'https://i.wolves.top/picgo/202410061408862.png'
---

<p style="color: aquamarine;text-align: center">UPDATED ON 2024-10-06 BY WOLVES</p>

## 目录
- [AI驱动NPC](#2024-10-06-14-08-00)

## 2024-10-06 14:08:00
After <span style="color: red">video</span> : [我让六个AI合租，居然出了个海王](https://www.bilibili.com/video/BV1MkxeeYEEb)

This video is based on the <span style="color: purple">paper</span> [https://arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442), which is called `Generative Agents: Interactive Simulacra of Human Behavior`

> feeling

- 将游戏中的NPC 设置为AI，接入语言模型，让语言模型模拟其生活，并赋予其一定的情感，可以显著提高游戏的沉浸感，甚至可以剖析现实世界的人类心理。
- 通过`prompt`可以改变AI的行为，甚至可以改变AI的情感，但是AI的情感是有限的，并不能像人类一样复杂，因此并不能完全模拟出人类的心理。并且很明显他们同化的速度极快，并且没有领地意识。

> thinking

- 现在的AI(LLM),<span style="color: green">通常以积极的方式进行回答</span>，即使输入了错误的信息，也会以积极的方式进行回答，而在真实世界中，人类会根据自身性格，当前的情绪状态，<span style="color: blue">对当前的问题进行评价</span>，并调整后续的回答，甚至拒绝回答，因此单纯的接入当前的LLM所模拟的实验也会趋向于积极的一面，而不能模拟出人类心理的复杂性。在这种情况下，NPC相互同化速度快。
- 关于领地意识，凸显出这个模拟实验的一些`bug`: <span style="color: red">现实规则</span>非常不完善，人类是有私有空间领地意识的，不会轻易的去他人的领地（如房间），在视频的项目中，AI进入他人的空间，会极大的加快同化速度，因此在设计阶段应该更多的增加现实世界规则以及AI的提示词，来完善这个实验。

> solution

- 首先，需要一个<span style="color: green">完善的现实世界规则</span>，如领地意识，私有空间，基本物理状态等。并且环境中应存在现实世界规则的提示词，来影响AI的行为。并且可以生成一些特殊事件，让ai进行特殊的反应。
- 其次，应该训练一个<span style="color: blue">特有的模型</span>，这个模型应该是可以模拟出人类心理复杂性的模型，可以根据状态进行优劣反应，而不是一味迎合回复。而且对于每个ai，他的提示词，性格，情感，记忆，交流，决策，行动都可以根据当前需求进行迭代。
    - 训练集优化，这个训练集应该拥有人物在进行对话时所处的环境位置、近期发生过什么(可以影响情绪)的大事件、与什么样的(不同关系)人进行的对话、人物接下来要做什么、以及说话风格等。<span style="color: gray">PS:这个训练集有点难嗷</span>
- 最后，需要一个<span style="color: red">完善的奖励机制</span>，来激励AI做出符合现实世界规则的行为，符合自我长期规划的行为，而非无目的性的进行活动。比如，增加金钱系统等。
