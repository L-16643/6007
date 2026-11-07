# 基于 OpenBandit 的离线策略评估（OPE）实证研究

强化学习课程项目。基于日本时尚电商 ZOZOTOWN 开源的 [Open Bandit Dataset](https://research.zozo.com/data_release/open_bandit_dataset.zip)（OBD，2690 万行真实推荐日志），训练上下文 Bandit 策略（LinUCB、LogisticTS），并系统评测 DM、IPS、SNIPS、DR 等离线策略评估估计器的偏差、方差与排序稳定性。

OBD 同时记录了**全随机（Random）**与 **Bernoulli Thompson Sampling（BTS）** 两个线上策略的日志，前者让我们能算出新策略的真实线上回报（Ground Truth），作为评估各估计器的标尺。

## 目录结构

```
.
├── README.md
├── config.yaml              # 全部可调参数（路径、切分比例、特征、置信水平）
├── pyproject.toml / uv.lock # 依赖（uv 管理，Python 3.11）
├── ope/                     # 项目代码包
│   └── data/                # 数据流水线 & Ground Truth（成员 B）
│       ├── io.py            #   常量、下载、读取、清洗、商品特征对齐
│       ├── features.py      #   用户上下文 / 商品特征编码
│       ├── dataset.py       #   构建流水线 + 加载接口 load_bandit_feedback
│       └── ground_truth.py  #   Ground Truth 与置信区间
├── scripts/                 # 命令行入口
│   ├── prepare_data.py      #   下载 + 清洗 + 切分 + 特征，一步完成
│   ├── ground_truth.py      #   计算 Ground Truth
│   └── example_policy.py    #   示例：生成策略分布并求其 Ground Truth
├── tests/                   # pytest 测试
├── docs/
│   └── data_pipeline.md     # 数据流水线设计细节与结果
├── results/ground_truth/    # Ground Truth 结果（csv / json）
└── data/                    # 数据集与处理后的数组（不进 git，脚本自动生成）
```

后续模块建议放在 `ope/` 下，与 `ope/data/` 并列，例如 `ope/policies/`（成员 C）、`ope/estimators/`（成员 D）、`ope/experiments/`（成员 E）。

## 快速开始

需要安装 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync                                        # 创建虚拟环境并安装依赖
```

```bash
uv run python scripts/prepare_data.py --sample # 先用官方 1 万行样例跑通（几秒）
```

```bash
uv run python scripts/prepare_data.py          # 全量数据：下载约 413 MB，构建约 2 分钟，输出约 11 GB
```

```bash
uv run python scripts/ground_truth.py          # 计算 Ground Truth -> results/ground_truth/
```

```bash
uv run pytest                                  # 运行测试
```

## 组员如何使用数据

```python
from ope.data import load_bandit_feedback

bf = load_bandit_feedback("all", policy="random", split="train")
bf["context"]         # (n, d_x)  用户上下文特征 X
bf["action"]          # (n,)      推荐的商品 a
bf["position"]        # (n,)      推荐位 0/1/2
bf["reward"]          # (n,)      是否点击 r
bf["pscore"]          # (n,)      行为策略倾向得分 π_b(a|x)
bf["action_context"]  # (n_actions, d_a) 商品特征
```

- `campaign`：`all`（80 个商品）、`men`（34）、`women`（46）
- `policy`：`random`（均匀随机）、`bts`（Bernoulli TS）
- `split`：`train`（时间上前 70%）、`test`（后 30%）

返回字典的键与 obp 库的 `bandit_feedback` 一致，可直接传给 obp 估计器。

**建议的实验协议**

| 用途 | 数据 |
|---|---|
| 训练 LinUCB / LogisticTS（成员 C） | `random/train` |
| 拟合奖励模型 q̂(x,a)（成员 D） | `random/train` 或 `bts/train` |
| 运行 OPE 估计器（成员 D、E） | `bts/test`：倾向得分极不均匀，最考验估计器 |
| 新策略的 Ground Truth | `random/test`：`scripts/ground_truth.py --action-dist` |

成员 C 的策略输出格式：在 `random/test` 的 `context` 上逐行计算 π_e(a|x)，保存为形状 `(n, n_actions, 3)` 或 `(n, n_actions)` 的 `.npy`，然后：

```bash
uv run python scripts/ground_truth.py --campaign all --action-dist linucb=outputs/linucb_all.npy
```

格式模板见 [scripts/example_policy.py](scripts/example_policy.py)。

## 当前结果

两个线上策略在测试集上的真实点击率（95% Wilson 置信区间）：

| campaign | Random | Bernoulli TS | BTS 相对 Random |
|---|---:|---:|---:|
| all | 0.347% [0.330%, 0.366%] | 0.491% [0.484%, 0.498%] | +41.4% |
| men | 0.541% [0.503%, 0.581%] | 0.665% [0.650%, 0.679%] | +22.9% |
| women | 0.487% [0.461%, 0.515%] | 0.637% [0.627%, 0.648%] | +30.7% |

BTS 日志的最小倾向得分为 1e-6，单个样本的重要性权重最高可达 10⁶，这正是 IPS 方差发散、需要 SNIPS / 权重截断的原因。更多细节见 [docs/data_pipeline.md](docs/data_pipeline.md)。

## 分工与进度

| 成员 | 角色 | 交付物 | 状态 |
|---|---|---|---|
| A | 组长 / 架构 | 开题报告、实验架构说明书、答辩主稿 | 进行中 |
| B | 数据流水线 | 数据加载器 `ope/data/`、Ground Truth 脚本 | ✅ 已完成 |
| C | Bandit 策略算法 | LinUCB、LogisticTS 训练代码，策略动作分布（.npy） | 待开始 |
| D | OPE 评估引擎 | DM / IPS / SNIPS / DR 估计器、MSE 自动化脚本 | 待开始 |
| E | 敏感度实验与可视化 | 截断敏感度实验、偏差-方差图表、答辩 Poster | 待开始 |

## 参考

Saito, Y., Aihara, S., Matsutani, M., & Narita, Y. (2020). *A Large-scale Open Dataset for Bandit Algorithms.* arXiv:2008.07146.
