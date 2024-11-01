## 3. 动作选择
### 3.1 多臂老虎机 （Multi-Armed Bandits)
<img src="bandits.png" alt="bandit" width="600"/>

让我们回到强化学习的理想模型：多臂老虎机（Multi-Armed Bandits）。在多臂老虎机问题中，我们有$A$个臂，每个臂的奖励服从某个固定的分布（为了简化起见，我们假设均为伯努利分布，第$a$个臂的奖励均值为未知数$p_a$），我们的目标是找到一个策略，使得累积奖励最大化。我们面临的困难既是所谓的Exploration-Exploitation Dilemma：我们需要在已知的臂中选择估算奖励最高的臂（Exploitation），同时也需要探索其他臂，以便更好地估计其奖励（Exploration）。下面，我们介绍解决这个问题的经典算法：UCB


### 3.2 Upper Confidence Bound (UCB)
首先，我们回顾下霍夫丁不等式（Hoeffding's inequality），令$X_1, X_2, \cdots, X_n$为独立同分布的随机变量, $S_n = X_1 + \cdots + X_n$，且$X_i \in [0, 1]$，则对任意$\epsilon > 0$，有：
$$
\text{Pr}(|\frac{S_n}{n}- \mathbb{E}[X]| \geq \epsilon) \leq 2e^{-2n\epsilon^2}
$$
即：
$$
\text{Pr}(|\mathbb{E}[X] - \frac{S_n}{n}| \leq \epsilon) \geq 1 - 2e^{-2n\epsilon^2}
$$
令$\gamma = 2e^{-2n\epsilon^2}$，则$\epsilon = \sqrt{\frac{\log(1/\gamma)}{2n}}$，$\mathbb{E}[X]$的置信区间 （$1-\gamma$概率）为：
$$
[\frac{S_n}{n} - \sqrt{\frac{\log(1/\gamma)}{2n}}, \frac{S_n}{n} + \sqrt{\frac{\log(1/\gamma)}{2n}}]
$$
代入多臂老虎机，设我们使用了第$a$个臂$N(a)=n$次，$p_a$的估计值即为$\frac{S_n}{n}$，记为$Q(a)$，所有臂的使用次数为$T$，令$1/\gamma = T^\alpha$, 则$p_a$的置信区间为：
$$
[Q(a) - \sqrt{\frac{\alpha\log(T)}{2N(a)}}, Q(a) + \sqrt{\frac{\alpha\log(T)}{2N(a)}}]
$$
UCB算法选择置信区间上界最大的臂，即：
$$
a^* 
= \arg\max_a \left\{Q(a) + \sqrt{\frac{\alpha\log(T)}{2N(a)}}\right\}
$$
可以看出，UCB算法在面对不确定性时采用了一种乐观的策略（以置信区间的上界作为选择依据），具体来讲，$Q(a)$表示利用exploitation的部分，$\sqrt{\frac{\alpha\log(T)}{2N(a)}}$表示探索Exploration的部分，使用次数$N(a)$越小则这项越大，$\alpha$是超参数来权衡两者。

可以证明，UCB算法的后悔界（Regret Bound）为$O(\log(T))$，即在$T$次操作后，UCB算法与上帝视角最优累积奖励（总是选择$p_a$最大的那个臂）之间的差距期望不会超过$O(\log(T))$，为理论最优。

### 3.3 Predictor Upper Confidence Bound for Trees (PUCT)

将UCB算法应用到树上并引入策略$P(s,a)$，即，在节点状态$s$下选择动作$a$的概率为$P(s,a)$，用以提升探索的效率，我们得到改进版的PUCT变体：
$$
\text{PUCT}(s, a) = Q(s, a) + c_{puct}  P(s, a)  \frac{\sqrt{\Sigma_b N(s, b)}}{1 + N(s, a)}
$$


# References
- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815)