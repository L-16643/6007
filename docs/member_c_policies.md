# 成员 C · 上下文 Bandit 策略

## 交接与复现

训练脚本默认把概率数组写到 `outputs/policies/`。概率数组保存在本地，仓库收录全量运行清单，便于复现和核对行序。[All](../results/policies/all/manifest.json) · [Men](../results/policies/men/manifest.json) · [Women](../results/policies/women/manifest.json)

每个 campaign 的本地输出目录包含以下文件。

| 文件 | 对应输入与用途 |
|---|---|
| `linucb_random_test.npy`、`logisticts_random_test.npy` | Random/test，供随机日志 IPS 参考值使用 |
| `linucb_bts_test.npy`、`logisticts_bts_test.npy` | BTS/test，供组员运行 OPE |
| `manifest.json` | 数据划分、训练规模、超参数、输出形状、行序摘要、IPS 参考值及诊断量 |

数组为 `float32`，形状为 `(n_rounds, n_actions, 3)`。`dist[i, a, p]` 是对应测试集第 `i` 行对商品 ID `a` 在推荐位 `p` 的选择概率，位置从 0 起算，每个位置沿动作轴求和为 1。数组必须与同 campaign、同行为策略的测试日志配对。清单的 `row_keys_sha256` 是该日志的 `(timestamp, action, position)` 三列以原顺序、小端 64 位整数、按行拼接后计算的 SHA-256。

```python
import numpy as np
from ope.data import load_bandit_feedback

bf = load_bandit_feedback("men", "bts", "test", mmap=True)
dist = np.load("outputs/policies/men/logisticts_bts_test.npy", mmap_mode="r")
assert dist.shape == (bf["n_rounds"], bf["n_actions"], bf["len_list"])
```

在 `6007` 项目根目录运行以下命令。若成员 B 的处理后数据已经准备好，可以跳过 `scripts.prepare_data`。已有结果时可用 `--out-dir outputs/policies/reproduction` 写到新目录。

```bash
uv sync --locked
uv run python -m scripts.prepare_data
uv run python -m scripts.train_policies
```

`--campaign men` 可只跑 Men，`--test-logs random` 可只导出 Random/test，`--help` 会列出超参数。脚本按块评分，把重复上下文的概率恢复到原始行序。每个 campaign 的文件在暂存目录写完并通过检查后一次性发布。

以下命令用官方样例验证训练和导出。

```bash
uv run python -m scripts.prepare_data --sample
uv run python -m scripts.train_policies --campaign men --processed-dir data/processed_sample \
    --out-dir outputs/policies/sample
uv run pytest
```

## 策略定义

每个 campaign 在自身的 Random/train 上离线拟合，使用成员 B 编码后的完整用户上下文和截距。对每个推荐位、商品分别估计参数，每条训练行只更新已展示商品和推荐位的模型。拟合后冻结参数，按两个测试日志的上下文分别导出概率。推荐位的输出为各自的边际分布。

**LinUCB。** 设$$z=[x;1]$$，按组累计$$A=\lambda I+\sum_i z_i z_i^\top$$、$$b=\sum_i z_i r_i$$。用 Cholesky 求解$$\hat\theta=A^{-1}b$$，然后按$$z^\top\hat\theta+\alpha\sqrt{z^\top A^{-1}z}$$ 取最高分。胜出商品的概率为 1，平分时选较小商品 ID。固定$$\lambda=1$$、$$\alpha=0.02$$。[Li 等，LinUCB](https://arxiv.org/abs/1003.0146)

**LogisticTS。** 按组拟合 Bernoulli 逻辑回归，以精度 1 的各向同性高斯为先验。截距先验均值为该推荐位的训练对数赔率 `log((点击数 + 0.5) / (未点击数 + 0.5))`，其余均值为 0。求得 MAP 后，用负对数后验 Hessian 的逆构造 Laplace 高斯近似。每组固定抽取 64 组参数，种子为 12345。每次比较各商品的 logit，胜出频率即动作概率。两种日志复用同一组抽样，概率分辨率为 1/64。[Chapelle 和 Li，Thompson Sampling](https://proceedings.neurips.cc/paper_files/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html)

## 本次全量运行

使用 ZOZO Open Bandit Dataset 1.0 及项目默认的按时间划分。官方 ZIP 的 SHA-256 为
`e8ec18196582a5937381a1776382ca940689b90a18d2dcd1fb635be6df614d78`。
运行环境为 Python 3.13.2、NumPy 2.5.4、SciPy 1.18.1、pandas 3.1.1 和 PyArrow 26.0.1。[ZOZO 官方归档](https://research.zozo.com/data_release/open_bandit_dataset.zip)

| Campaign | Random/train | Random/test | BTS/test | 商品数 |
|---|---:|---:|---:|---:|
| All | 961,915 | 412,250 | 3,706,504 | 80 |
| Men | 317,008 | 135,860 | 1,223,053 | 34 |
| Women | 605,114 | 259,334 | 2,329,202 | 46 |

以下参考值由成员 B 的 `policy_value_from_random_logs` 在 Random/test 上计算。区间为该接口输出的 95% 正态近似区间，ESS 是重要性加权后的有效样本量。确定性 LinUCB 的有效样本量较小。各结果需结合区间解读。

| Campaign | 策略 | Random 日志 CTR | 策略 IPS 参考值及区间 | ESS |
|---|---|---:|---:|---:|
| All | LinUCB | 0.347% | 0.563% [0.358%, 0.768%] | 5,068 |
| All | LogisticTS | 0.347% | 0.468% [0.394%, 0.541%] | 30,020 |
| Men | LinUCB | 0.541% | 0.400% [0.204%, 0.597%] | 4,158 |
| Men | LogisticTS | 0.541% | 0.548% [0.449%, 0.647%] | 19,487 |
| Women | LinUCB | 0.487% | 0.745% [0.520%, 0.970%] | 5,658 |
| Women | LogisticTS | 0.487% | 0.616% [0.510%, 0.721%] | 22,135 |

官方 Men 样例的训练和导出通过，项目测试共 13 项通过。12 份全量文件重新打开后，逐块核对了形状、数值范围和按动作归一化。每份文件另抽查 256 行，与重新拟合的策略预测对照。两个测试集的行序摘要及 Random/test 的 IPS 参考值均重新计算，并与清单一致。

## 修改记录

| 日期 | 路径 | 内容 |
|---|---|---|
| 2026-11-10 | `ope/policies/__init__.py` | 实现位置感知的 LinUCB、LogisticTS 和冻结后的概率接口 |
| 2026-11-10 | `scripts/train_policies.py` | 复用 B 的加载器，生成两类测试日志的概率、行序摘要和 IPS 参考值 |
| 2026-11-10 | `tests/test_policies.py` | 验证置信项、后验抽样投票、固定种子、冷启动和行序 |
| 2026-11-10 | `outputs/policies/`、`outputs/.gitignore` | 本机保存 12 份概率数组与 3 份清单，由训练脚本生成，排除在 Git 版本管理之外 |
| 2026-11-10 | `docs/member_c_policies.md` | 记录接口、算法、复现步骤、实测结果与文件变更 |
| 2026-11-10 | `results/policies/` | 保存三组全量运行清单，记录超参数、数据行序摘要与 IPS 参考值 |
