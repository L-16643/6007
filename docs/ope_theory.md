# OPE 理论学习笔记（Week 1）

> 成员 D · Off-Policy Evaluation（OPE）评估引擎  
> 目标：理解 Contextual Bandit 的 OPE 问题，能独立解释 DM / IPS / SNIPS / DR，并跑通最小实验。

## 1. 问题设定

Contextual Bandit 的一轮交互可以写成：

1. 环境给出上下文 <code>x ∈ R^d</code>（用户、商品、位置等特征）；
2. 策略根据上下文选择一个动作 <code>a ∈ {1, ..., K}</code>（推荐哪个商品）；
3. 环境给出奖励 <code>r ∈ {0, 1}</code>（是否点击）；
4. 日志系统记录 <code>(x, a, r)</code>，同时记录行为策略选择该动作的概率。

推荐位 <code>l ∈ {1, ..., L}</code> 也可以作为上下文/动作条件的一部分。本项目的 OBD 数据有 3 个推荐位，因此统一使用位置感知的记号：

~~~text
q(x, a, l) = E[r | x, a, l]
~~~

目标策略 <code>π_e(a | x, l)</code> 的**真实策略价值**定义为：

~~~text
V(π_e) = E_{x,l} [ Σ_a π_e(a | x, l) · q(x, a, l) ]
~~~

离线评估只拿到行为策略产生的历史日志，不能真的让新策略上线，因此需要从日志中估计 <code>V(π_e)</code>。

## 2. 行为策略、目标策略与倾向得分

- **Behavior Policy（行为策略）** <code>π_b</code>：真正产生日志的策略。OBD 中有全随机 Random 和 Bernoulli Thompson Sampling（BTS）两种。
- **Evaluation Policy（目标策略）** <code>π_e</code>：想评估但尚未上线的新策略，例如 LinUCB、LogisticTS。
- **Propensity Score（倾向得分）**：日志中实际被选中的动作，在行为策略下的概率

~~~text
p_i = π_b(a_i | x_i, l_i)
~~~

OBD 的 <code>pscore</code> 就是 <code>p_i</code>。BTS 日志中最小倾向得分可到 <code>1e-6</code>，这意味着单个样本的重要性权重可能达到 <code>10^6</code>，是 IPS 方差爆炸的直接原因。

### 2.1 关键假设

1. **Unconfoundedness / Ignorability**：给定 <code>(x, l)</code> 后，动作与潜在奖励独立；日志中的动作只由行为策略决定。
2. **Positivity / Overlap**：只要 <code>π_e(a|x,l) > 0</code>，就必须有 <code>π_b(a|x,l) > 0</code>。否则该动作在日志中永远观测不到，无法用重要性加权修正。
3. **Known propensity**：本项目的 <code>pscore</code> 由数据流水线提供，视为已知。
4. **Bounded reward**：点击奖励在 <code>[0, 1]</code>，便于讨论方差和置信区间。

## 3. 重要性权重

对第 <code>i</code> 条日志，定义重要性权重：

~~~text
w_i = π_e(a_i | x_i, l_i) / π_b(a_i | x_i, l_i)
~~~

它把“行为策略下观测到的奖励”重新加权成“目标策略下的期望奖励”。如果目标策略在某条日志上更可能选择该动作，<code>w_i > 1</code>；否则 <code>w_i < 1</code>。

重要性加权的两个直接问题：

- **高方差**：当 <code>π_b</code> 很小而 <code>π_e</code> 不小，<code>w_i</code> 会非常大；
- **数值不稳定**：少量极端权重会主导均值，甚至让估计值超出 <code>[0, 1]</code>。

常见处理是 **Clipping**：

~~~text
w_i^clip = min(w_i, c)
~~~

其中 <code>c</code> 是截断阈值。Clipping 降低方差，但会引入偏差；阈值敏感度是成员 E 的后续实验内容。

## 4. 四种 OPE 估计器

设日志共 <code>n</code> 条，<code>q̂(x,a,l)</code> 是奖励预测模型（点击概率模型）。

### 4.1 Direct Method（DM）

DM 直接用奖励模型预测每个动作的期望奖励，再按目标策略加权：

~~~text
V_DM = (1/n) Σ_i Σ_a π_e(a | x_i, l_i) · q̂(x_i, a, l_i)
~~~

- **优点**：不依赖倾向得分，方差通常较低；即使目标策略与行为策略差异很大也能算。
- **缺点**：如果 <code>q̂</code> 有模型偏差，DM 就有系统性偏差；它没有利用日志中观测到的奖励做修正。
- **关键**：奖励模型质量决定 DM 的上限。

### 4.2 Inverse Probability Weighting（IPS）

IPS 用重要性权重修正行为策略和目标策略的分布差异：

~~~text
V_IPS = (1/n) Σ_i w_i · r_i
~~~

- **优点**：只要倾向得分正确且满足 overlap，就是无偏的；不需要奖励模型。
- **缺点**：方差可能极大，尤其是 <code>π_b</code> 很小的 BTS 日志；对倾向得分估计误差敏感。
- **特例**：当 <code>π_e = π_b</code> 时，<code>w_i = 1</code>，IPS 退化为日志奖励均值。

### 4.3 Self-Normalized IPS（SNIPS）

SNIPS 对权重做自归一化：

~~~text
V_SNIPS = Σ_i w_i · r_i / Σ_i w_i
~~~

- **优点**：分母与分子共享同一个权重尺度，显著降低方差；对权重整体缩放不敏感。
- **缺点**：有限样本下不再严格无偏，但通常一致；当 <code>Σ w_i</code> 很小时仍不稳定。
- **直觉**：它把 IPS 的“绝对权重”变成“相对权重”，类似加权平均。

### 4.4 Doubly Robust（DR）

DR 把 DM 和 IPS 结合起来：

~~~text
V_DR = (1/n) Σ_i [ Σ_a π_e(a | x_i, l_i) · q̂(x_i, a, l_i)
                  + w_i · (r_i - q̂(x_i, a_i, l_i)) ]
~~~

- 第一项是 DM 的预测值；
- 第二项是用 IPS 加权的**残差修正**：如果模型预测准，残差接近 0，DR 接近 DM；如果模型有偏，但权重正确，残差项会把偏差修正回来。
- **Double Robustness**：在理想条件下，只要 <code>q̂</code> 正确**或**重要性权重正确，DR 就是一致的。
- **注意**：实际中 <code>q̂</code> 和 <code>π_b</code> 都是估计/近似的，double robustness 不是无限保证；q 模型太差时 DR 也可能不稳定。

### 4.5 对比总结

| 估计器 | 是否用 q̂ | 是否用倾向得分 | 主要偏差来源 | 主要方差来源 |
|---|---|---|---|---|
| DM | 是 | 否 | 奖励模型设定错误 | 模型预测波动 |
| IPS | 否 | 是 | 倾向得分错误 / 无 overlap | 极端重要性权重 |
| SNIPS | 否 | 是 | 有限样本自归一化偏差 | 权重分布 |
| DR | 是 | 是 | q̂ 或权重至少一个错 | 残差 × 权重 |

Week 1 的目标是**理解并跑通**，不是下结论说哪个最好。单次实验的绝对误差同时包含数据噪声和模型误差，估计器的偏差/方差需要 Week 3 的重复实验与 Bootstrap 才能可靠比较。

## 5. 奖励预测模型与 Cross-Fitting

DM 和 DR 都需要 <code>q̂(x,a,l) ≈ P(r=1 | x,a,l)</code>。Week 1 的最小实现：

1. 对每个 <code>(action, position)</code> 组合分别训练一个 Logistic Regression；
2. 输入是上下文 <code>x</code>，标签是点击 <code>r</code>；
3. 用 3 折 **cross-fitting**：每条样本的 <code>q̂</code> 都由“没有见过该样本”的模型预测；
4. 如果某个训练折里该动作/位置全是 0 或全是 1，用训练折点击率做平滑回退，避免 <code>predict_proba</code> 失败；
5. 输出 <code>q̂</code> 的形状为 <code>(n_rounds, n_actions, len_list)</code>。

Cross-fitting 的作用是减少“用同一批数据既训练 q 又评估 q”带来的过拟合偏差。它不是万能的：如果真实 <code>q</code> 不是 logistic 形式，模型仍然会 misspecify；这正是后续比较 DM/DR 对模型质量敏感度的动机。

## 6. OBP 的数据与 API 契约

Week 1 使用 OBP 的 <code>SyntheticBanditDataset</code> 生成带已知 <code>expected_reward</code> 的模拟日志。OBP 0.4.1 的关键字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| <code>context</code> | <code>(n, d_x)</code> | 上下文 <code>x</code> |
| <code>action</code> | <code>(n,)</code> | 日志中被选中的动作 |
| <code>reward</code> | <code>(n,)</code> | 点击奖励 |
| <code>pscore</code> | <code>(n,)</code> | <code>π_b(a_i|x_i)</code> |
| <code>expected_reward</code> | <code>(n, K)</code> | 真实 <code>q(x,a)</code>，仅模拟数据有 |
| <code>position</code> | <code>(n,)</code> | OBP 0.4.1 的 synthetic 数据返回 <code>None</code>，本项目适配为全 0 |
| <code>n_actions</code> | <code>int</code> | 动作数 <code>K</code> |

OBP 估计器的输入约定：

- <code>action_dist</code>：<code>(n, K, L)</code>，每个位置对动作求和为 1；
- <code>estimated_rewards_by_reg_model</code>：<code>(n, K, L)</code>；
- <code>pscore</code>：<code>(n,)</code>；
- <code>position</code>：<code>(n,)</code>，默认全 0。

对应 OBP 类：

~~~python
from obp.ope import (
    DirectMethod,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    DoublyRobust,
)
~~~

本项目的 <code>ope.estimators.evaluate</code> 实现了同样的公式，并在 <code>tests/test_estimators.py</code> 中与 OBP 官方实现逐项对比。Week 1 的对比使用**同一个 q̂** 喂给两边，因此比较的是估计器算术，而不是 OBP 的 <code>RegressionModel</code> 拟合质量。

## 7. Week 1 最小实验设计

- 数据：OBP <code>SyntheticBanditDataset</code>，自定义 context-dependent reward / behavior policy；
- 规模：5000 条日志；
- 随机种子：42；
- 奖励模型：3 折 cross-fitting，每个动作/位置一个 Logistic Regression；
- 目标策略：
  - <code>behavior</code>：与行为策略相同，用于 sanity check；
  - <code>uniform</code>：均匀策略；
  - <code>contextual</code>：context-dependent softmax；
  - <code>sharp</code>：低温 softmax，接近 deterministic，分布偏移最大；
- 指标：真实值 <code>V(π_e)</code>、绝对误差、平方误差、与 OBP 官方估计的最大绝对差。

## 8. 本周期待与边界

Week 1 已完成：

- 理解并解释 DM / IPS / SNIPS / DR；
- 在 OBP 模拟数据上跑通四种估计器；
- 与官方 OBP 实现做数值对照；
- 输出结果 CSV/JSON 与实验说明。

Week 1 **不**完成：

- 真实 OBD 日志上的 benchmark（Week 3）；
- 多次重复实验的 MSE / Bias / Variance / Bootstrap CI（Week 3）；
- 完整的统一估计器接口、Clipping 敏感度、Switch-DR（Week 2–4）；
- 对 LinUCB / LogisticTS 的最终评估（依赖成员 C）。

## 9. 参考资料

- Saito, Y., Aihara, S., Matsutani, M., & Narita, Y. (2020). *A Large-scale Open Dataset for Bandit Algorithms.* arXiv:2008.07146.
- Dudík, M., Erhan, D., Langford, J., & Li, L. (2014). *Doubly Robust Policy Evaluation and Optimization.*
- Swaminathan, A., & Joachims, T. (2015). *The Self-Normalized Estimator for Counterfactual Learning.*
- Open Bandit Pipeline: <https://github.com/st-tech/zr-obp>
