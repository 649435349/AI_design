README
===========================
****
### Author:冯宇飞
### E-mail:649435349@qq.com
### Wechat:fyf649435349
****

## What is it?
```
仅作校招参考使用……
这是我在某游戏公司实习的时候写的部分代码，最后没有上线，仅作为参考使用，大部分由我编写，如果有侵权请通知我。
背景是在某一款游戏的竞技场，有三个职业：战士（主近战），牧师（主治疗）和猎人（主远程），每个人有5个技能+普攻。
对方是用BT树写的AI，而我的任务是研发新的算法，效果比现有的优秀即可。
```
   
## What is used and what is the performance?
```
全部用Python写成，Python也是我最喜欢的语言。
为了快速开发实现各种修改和想法，框架采用BG是Tensorflow的Keras，也有自己手写的神经网络。
网络通信部分才用了socket编程和多线程threading模式。
我用的所有方法，全部采用了2000到现在（2017）的强化学习方法（DRL），如Policy Gradient Networks,Deep Q-Network
,Double DQN,Dueling DQN,A3c,Trop(借鉴了[https://github.com/joschu/modular_rl](https://github.com/joschu/modular_rl))等。
结果是PG和Dueling DQN获得了比较好的效果，每个职业的对战均能获得0.7左右的胜率，合理的技能搭配之后能达到0.9左右
遗憾的是，由于实习时间有限，我未能研究完其他算法，由于技术平台限制，不能够使人物边移动边放技能，效果始终差强人意。
```

## The Structure of the project?
```
简单介绍几个比较重要的文件夹。
`./_remoteai/resources` `./_remoteai/model`有训练好的模型参数
`./_remoteai/bin/algo`有大量现成的模型代码，`./ai.py`是手写的PG网络，`./ai_pg.py`是手写的PG网络2，
`./ai_dqn_fyf.py`是DQN代码，`./ai_dueling_dqn_fyf.py`是Dueling DQN代码,`./ai_pn_fyf`是固定顺序的尝试，
`./ai_mc.py`是蒙特卡洛随机法，`./ai_trpo.py`是Trpo算法的实现（A3C算法在地震中消失了……当然A3C很简单，就是融合了PG和DQN，
然后异步获取数据，不用存储过往数据）`./_remoteai/client`是客户端请求代码。
其他并不是我写的了，仅做参考。
```
