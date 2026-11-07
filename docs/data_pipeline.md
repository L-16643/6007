# 数据流水线 & Ground Truth（成员 B）

本模块负责：OpenBandit 数据集（OBD）的**下载 → 解析 → 清洗 → 时间切分 → 特征工程**，并输出**全随机基线 Ground Truth**。
其他成员只需要调用 `load_bandit_feedback(...)` 就能拿到对齐好的 `X / A / R / π_b`。

> 安装、运行命令和目录结构见项目根目录的 [README.md](../README.md)，本文只讲设计细节与结果。

| 模块 | 作用 |
|---|---|
| `ope/data/io.py` | 数据集常量、下载、读取原始 csv、清洗校验、倾向得分统计、商品特征对齐 |
| `ope/data/features.py` | `ContextEncoder`（用户上下文 X）与 `build_action_context`（商品特征） |
| `ope/data/dataset.py` | 端到端构建 + 给组员用的加载接口 `load_bandit_feedback` |
| `ope/data/ground_truth.py` | Ground Truth 计算（on-policy 值、随机日志 IPS、置信区间） |

## 1. 数据处理细节

### 1.1 清洗规则（`clean_logs`）
- 删除关键列 / 特征缺失行、完全重复行；
- 校验 `item_id ∈ [0, n_items)`、`position ∈ {1,2,3}`、`click ∈ {0,1}`、`propensity_score ∈ (0,1]`、亲和度非负；
- `position` 转为 0 起始（与 OBP 一致），按时间戳稳定排序；
- 每一类被删除的行数、CTR、各推荐位 CTR、倾向得分分位数都写入 `meta.json["cleaning"]`。
- **不在数据层裁剪倾向得分**：裁剪（Clipping）是估计器层面的选择，留给成员 D/E 做敏感度实验。`meta.json` 里的 `pscore.min / max_inverse / frac_below_1e-3` 可直接用来说明 BTS 日志的权重爆炸风险。

### 1.2 发现的数据坑：商品类别特征哈希不一致
`random/` 与 `bts/` 两个文件夹里的 `item_context.csv`，数值特征 `item_feature_0` 完全相同，但类别特征 `item_feature_1..3` 的哈希字符串**用了不同的盐**。直接拼接会让同一商品在两个策略下特征不同。
处理：以 `random` 为标准，校验两边对商品的划分是一一对应（双射）后，统一使用标准标签（`align_item_context`）。用户特征 `user_feature_0..3` 两边哈希一致，无需处理。

### 1.3 切分
按时间顺序切分（默认 train 70% / test 30%，可配 val），与 OBP 的 time-series split 一致，避免未来信息泄漏。random 和 bts 各自独立切分。

### 1.4 特征工程（`ContextEncoder`）
上下文向量 X 的列顺序（名称见 `meta.json["context_feature_names"]`）：

| 块 | 内容 | 处理 |
|---|---|---|
| 用户类别特征 | `user_feature_0..3` | one-hot；出现次数 < 10 的类别 + 测试集新类别 → `__other__` |
| 用户-商品亲和度 | `user-item_affinity_0..K-1`（历史交互计数，长尾） | `log1p` 后 z-score |
| 亲和度统计 | log1p(总亲和度)、有亲和度的商品数 | z-score |

- 编码器**只在 random 训练集上拟合**：用户上下文分布与日志策略无关，random/train 就是总体分布的无偏样本，且不碰测试集。
- 维度（全量数据）：all = 116，men = 69，women = 80（样例数据类别更少，维度会小一些，以 `meta.json["context_dim"]` 为准）。
- 商品特征 `action_context`：`item_feature_0`（数值）+ 三个类别特征 one-hot，两策略共用。

## 2. 输出格式 & 队友接口

```python
from ope.data import load_bandit_feedback, load_meta

bf = load_bandit_feedback("all", policy="random", split="train")   # policy: random|bts, split: train|test
bf["context"]         # (n, d_x) float32   用户上下文 X
bf["action"]          # (n,)     int64     推荐的商品 a
bf["position"]        # (n,)     int64     推荐位 0/1/2
bf["reward"]          # (n,)     int64     点击 r
bf["pscore"]          # (n,)     float64   行为策略倾向得分 π_b(a|x,pos)
bf["action_context"]  # (n_actions, d_a)   商品特征
bf["n_rounds"], bf["n_actions"], bf["len_list"]
```

字典键与 `obp.dataset.OpenBanditDataset.obtain_batch_bandit_feedback()` 一致，可直接喂给 obp 估计器。bts/all 训练集约 850 万行，可用 `mmap=True` 内存映射 `context`。

**建议的实验协议（供 C/D 参考）**

| 用途 | 数据 |
|---|---|
| 成员 C 训练 LinUCB / LogisticTS | `random/train`（无偏，无需倾向得分修正） |
| 成员 D 拟合奖励模型 q̂(x,a) | `random/train` 或对应行为日志的 train |
| 成员 D 跑 OPE（分布偏移场景） | `bts/test`（倾向得分极不均匀，最考验估计器） |
| 新策略 Ground Truth | `random/test`（见下） |
| 已部署策略 Ground Truth | 各策略自己的 `test` 在线 CTR |

## 3. Ground Truth 方法

**(1) 已部署策略的真实值（on-policy）**：OBD 同时记录了 Random 和 Bernoulli TS 两个线上策略。一个策略在其自身测试日志上的平均点击率就是其线上真实价值的无偏估计，用 Wilson 区间（适合 0.4% 量级的低 CTR）+ bootstrap 给出 95% 置信区间。
这就是 OBP 论文的标准评测：用 random 日志离线估计 V(bts)，与 bts 测试日志的真实 CTR 对比（反之亦然）。

**(2) 新策略（LinUCB、LogisticTS 等，从未上线）的真实值**：Random 策略是均匀的，π_b = 1/K 精确已知，重要性权重 π_e/π_b ≤ K 有界，因此在**留出的 random/test** 上做 IPS 是无偏且方差可控的：

  V_GT(π_e) = (1/n) Σ_i  π_e(a_i | x_i, pos_i) · K · r_i

置信区间：正态近似 + 精确 bootstrap（利用点击稀疏性的两阶段抽样，O(点击数)/次）。同时输出 SNIPS 值、平均权重（应≈1，用于检查 π_e 是否合法）、最大权重、有效样本量 ESS。

成员 C 交付策略后：

```bash
uv run python scripts/ground_truth.py --campaign all \
    --action-dist linucb=outputs/policies/all/linucb_random_test.npy \
                  logisticts=outputs/policies/all/logisticts_random_test.npy
```

`.npy` 要求：在 `load_bandit_feedback(campaign, "random", "test")["context"]` 上逐行计算，形状 `(n, n_actions, 3)` 或 `(n, n_actions)`，每行对动作求和为 1。也可以在 Python 里直接调用 `policy_value_from_random_logs(bf, action_dist)`。

输出：`results/ground_truth/ground_truth.csv|json`（值、标准误、CI、相对 Random 的提升）与 `ctr_by_position.csv`。

## 4. 全量数据结果（2019-11-24 ~ 2019-11-30，共 26.9M 行）

### 4.1 清洗与切分（`data/processed/data_summary.csv`）

| campaign | policy | train 行数 | test 行数 | test CTR | π_b 范围 | 清洗删除 |
|---|---|---:|---:|---:|---|---:|
| all | random | 961,915 | 412,250 | 0.347% | 恒为 1/80 | 162 |
| all | bts | 8,648,508 | 3,706,504 | 0.491% | 1e-6 ~ 0.954 | 2,188 |
| men | random | 317,008 | 135,860 | 0.541% | 恒为 1/34 | 81 |
| men | bts | 2,853,790 | 1,223,053 | 0.665% | 1e-6 ~ 0.725 | 884 |
| women | random | 605,114 | 259,334 | 0.487% | 恒为 1/46 | 137 |
| women | bts | 5,434,803 | 2,329,202 | 0.637% | 1e-6 ~ 0.963 | 1,492 |

- 删除的行全部是**完全重复行**（约 0.02%），没有缺失值或非法值。
- BTS 日志最小倾向得分为 1e-6，即单个样本的 IPS 权重因子最高可达 10⁶；all 中约 1% 的样本 π_b < 1e-3。这是 IPS 方差发散、需要做 SNIPS / Clipping 的直接数据证据。
- 全量构建约 2 分钟，输出约 11 GB。

### 4.2 Ground Truth（test 集，`results/ground_truth/ground_truth.csv`）

| campaign | 策略 | 真实 CTR V(π) | 95% CI (Wilson) | 相对 Random |
|---|---|---:|---|---:|
| all | Random | 0.003474 | [0.003299, 0.003658] | — |
| all | Bernoulli TS | 0.004910 | [0.004840, 0.004982] | +41.4% |
| men | Random | 0.005410 | [0.005034, 0.005814] | — |
| men | Bernoulli TS | 0.006646 | [0.006504, 0.006792] | +22.9% |
| women | Random | 0.004874 | [0.004613, 0.005149] | — |
| women | Bernoulli TS | 0.006373 | [0.006271, 0.006476] | +30.7% |

### 4.3 新策略 GT 通路验证（`scripts/example_policy.py`）

| campaign | 示例策略 | V_GT | 95% CI | 平均权重 | ESS |
|---|---|---:|---|---:|---:|
| men | uniform | 0.005410 | [0.005020, 0.005800] | 1.000 | 135,860 |
| men | popularity（按位置 CTR 的 softmax） | 0.005591 | [0.004956, 0.006227] | 1.016 | 44,636 |
| all | uniform | 0.003474 | [0.003294, 0.003653] | 1.000 | 412,250 |
| all | popularity | 0.004732 | [0.004344, 0.005119] | 1.004 | 148,565 |

uniform 策略的 GT 与 Random 的 on-policy CTR 完全一致（正确性检查）；即便是不看上下文的流行度策略，在 all 上也比 Random 高约 36%。这个脚本可以作为成员 C 输出 `action_dist` 的格式模板。
