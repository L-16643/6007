# Week 1 运行结果：最小 OPE 实验

> 成员 D · OPE 评估引擎  
> 日期：2026-11-08  
> 状态：Week 1 验收通过（模拟数据 + OBP 官方实现数值对照）

## 1. 环境

- conda 环境：<code>openbandit</code>
- Python：3.11.15
- Open Bandit Pipeline：<code>obp 0.4.1</code>
- NumPy：2.4.6
- scikit-learn：1.9.0
- 运行命令：

~~~bash
conda run -n openbandit python ope_demo.py
conda run -n openbandit python -m pytest -q
~~~

## 2. 实验设置

- 数据生成：OBP <code>SyntheticBanditDataset</code> + 自定义 context-dependent DGP；
- 样本量：5000 条日志；
- 随机种子：42；
- 动作数：3；
- 行为策略：context-dependent softmax，temperature = 1.15；
- 奖励函数：<code>q(x,a) = sigmoid(x W + c)</code>，已知真实值；
- 奖励模型：3 折 cross-fitting，每个 <code>(action, position)</code> 一个 Logistic Regression；
- 目标策略：
  - <code>behavior</code>：与行为策略相同；
  - <code>uniform</code>：均匀策略；
  - <code>contextual</code>：context-dependent softmax；
  - <code>sharp</code>：temperature = 0.35 的尖锐 softmax，分布偏移最大；
- 估计器：DM / IPS / SNIPS / DR；
- 官方对照：OBP 的 <code>DirectMethod</code>、<code>InverseProbabilityWeighting</code>、<code>SelfNormalizedInverseProbabilityWeighting</code>、<code>DoublyRobust</code>。

> 说明：为保证“估计器算术”可比，OBP 官方估计器接收的 <code>q̂</code> 与本地实现完全相同；没有使用 OBP 的 <code>RegressionModel</code> 重新拟合。OBP 0.4.1 的 synthetic 数据返回 <code>position=None</code>，实验中适配为全 0，并设置 <code>len_list=1</code>。

## 3. 结果

真实值 <code>V(π_e)</code> 与四种估计器的估计值（保留 6 位小数）：

| 目标策略 | 真实值 | DM | IPS | SNIPS | DR | 最小绝对误差 |
|---|---:|---:|---:|---:|---:|---|
| behavior | 0.575302 | 0.565053 | 0.564600 | 0.564600 | 0.565060 | DR（0.010242） |
| uniform | 0.513597 | 0.509119 | 0.507888 | 0.507753 | 0.508930 | DM（0.004478） |
| contextual | 0.374060 | 0.379136 | 0.379133 | 0.373865 | 0.375858 | SNIPS（0.000195） |
| sharp | 0.337179 | 0.345580 | 0.348665 | 0.341735 | 0.342332 | SNIPS（0.004556） |

绝对误差（<code>|estimate - truth|</code>）：

| 目标策略 | DM | IPS | SNIPS | DR |
|---|---:|---:|---:|---:|
| behavior | 1.02e-02 | 1.07e-02 | 1.07e-02 | 1.02e-02 |
| uniform | 4.48e-03 | 5.71e-03 | 5.84e-03 | 4.67e-03 |
| contextual | 5.08e-03 | 5.07e-03 | 1.95e-04 | 1.80e-03 |
| sharp | 8.40e-03 | 1.15e-02 | 4.56e-03 | 5.15e-03 |

### 3.1 与 OBP 官方实现的一致性

对 4 个目标策略 × 4 个估计器共 16 组结果：

- 本地实现与 OBP 官方实现的最大绝对差：<code>1.11e-16</code>；
- 绝大多数组合的差为 0，只有浮点求和顺序造成的 <code>1e-16</code> 量级差异；
- 说明 IPS / SNIPS / DM / DR 的公式、位置索引和权重计算与 OBP 0.4.1 一致。

### 3.2 初步观察

1. **behavior 目标**：目标策略等于行为策略时，IPS 的权重全为 1，估计值就是日志平均点击率；四种估计器都接近真实值，差异主要来自采样噪声和 q 模型误差。
2. **uniform 目标**：DM 和 DR 的误差较小。均匀策略与行为策略的覆盖较好，奖励模型也容易学。
3. **contextual / sharp 目标**：分布偏移变大，IPS 的方差问题更明显；SNIPS 的自归一化在这两次单次实验中误差最低。DR 介于 DM 和 SNIPS 之间，说明残差修正起了作用，但没有“自动最优”。
4. **没有 universally best estimator**：不同目标策略下排名会变化。单次实验不能据此判断偏差/方差，必须做重复实验和置信区间（Week 3）。

## 4. 可复现产物

- 演示脚本：<code>ope_demo.py</code>
- 原始结果（长表，含 ours / obp 两种 implementation）：<code>results/week1/ope_demo_results.csv</code>
- 汇总结果（配置、真实值、估计值、与 OBP 的最大差）：<code>results/week1/ope_demo_summary.json</code>
- 测试：<code>tests/test_estimators.py</code>、<code>tests/test_reward_model.py</code>
- 理论笔记：<code>docs/ope_theory.md</code>

重新生成：

~~~bash
conda run -n openbandit python ope_demo.py
~~~

不依赖 OBP 的本地回退：

~~~bash
python ope_demo.py --dataset local --no-save
~~~

## 5. Week 1 验收对照

| 交付物 | 状态 | 位置 |
|---|---|---|
| <code>ope_demo.py</code> | 完成 | 项目根目录 |
| OPE 理论学习笔记 | 完成 | <code>docs/ope_theory.md</code> |
| 四种估计器初步运行结果 | 完成 | <code>results/week1/</code> |
| 能解释四种估计器原理 | 完成 | 理论笔记第 4 节 |
| 能独立运行完整 OPE 示例 | 完成 | 上述命令，20 个测试通过 |

## 6. 局限性

- 当前是**模拟数据**，不是 OBD 真实日志；真实数据的稀疏点击、极端倾向得分和位置效应对估计器的影响要到 Week 3 验证。
- 仅运行了一个随机种子的一次实现；表中的绝对误差包含奖励采样噪声，不能直接当作估计器的偏差。
- 奖励模型是每个动作/位置独立的 Logistic Regression，且 DGP 与模型形式匹配；真实数据上可能出现模型 misspecification。
- 尚未实现 Bootstrap 置信区间、MSE / Bias / Variance、Clipping 敏感度和 Switch-DR；这些属于 Week 3–4。
- 目标策略目前是人工构造的 softmax，不是成员 C 的 LinUCB / LogisticTS 输出。
