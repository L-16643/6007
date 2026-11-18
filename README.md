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
│   ├── estimators.py        # OPE 估计器（成员 D，Week 1 最小实现）
│   ├── reward_model.py      # 交叉拟合奖励模型（成员 D）
│   └── data/                # 数据流水线 & Ground Truth（成员 B）
│       ├── io.py            #   常量、下载、读取、清洗、商品特征对齐
│       ├── features.py      #   用户上下文 / 商品特征编码
│       ├── dataset.py       #   构建流水线 + 加载接口 load_bandit_feedback
│       └── ground_truth.py  #   Ground Truth 与置信区间
├── ope_demo.py              # Week 1 最小 OPE 实验（成员 D）
├── scripts/                 # 命令行入口
│   ├── prepare_data.py      #   下载 + 清洗 + 切分 + 特征，一步完成
│   ├── ground_truth.py      #   计算 Ground Truth
│   └── example_policy.py    #   示例：生成策略分布并求其 Ground Truth
├── tests/                   # pytest 测试
│   ├── test_data.py
│   ├── test_estimators.py
│   └── test_reward_model.py
├── docs/
│   ├── data_pipeline.md     # 数据流水线设计细节与结果
│   ├── ope_theory.md        # OPE 理论学习笔记（成员 D）
│   └── week1_results.md     # Week 1 运行结果与验收
├── results/
│   ├── ground_truth/        # Ground Truth 结果（csv / json）
│   └── week1/               # OPE 最小实验结果（csv / json）
└── data/                    # 数据集与处理后的数组（不进 git，脚本自动生成）
```

后续模块建议放在 `ope/` 下，与 `ope/data/` 并列，例如 `ope/policies/`（成员 C）、`ope/experiments/`（成员 E）。成员 D 的 OPE 最小实现已放在 `ope/estimators.py`、`ope/reward_model.py` 与 `ope_demo.py`。

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

# OPE Week 1（需要 OBP；本机 conda 环境名为 openbandit）
conda run -n openbandit python ope_demo.py
conda run -n openbandit python -m pytest -q
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
| C | Bandit 策略算法 | LinUCB、LogisticTS 训练代码，策略动作分布（.npy）与[交接说明](docs/member_c_policies.md) | ✅ 已完成 |
| D | OPE 评估引擎 | DM / IPS / SNIPS / DR 估计器、[MSE 自动化脚本](scripts/evaluate_mse.py) | ✅ 已完成 |
| E | 敏感度实验与可视化 | 截断敏感度实验、偏差-方差图表、[实验报告](docs/member_e_experiments.md)、[最终 Poster](poster/poster.pdf) | ✅ 已完成 |

## 成员 D · OPE 最小实验（Week 1 完成）

Week 1 已交付 OPE 最小实验、理论学习笔记和可复现结果。实验使用 OBP 的 <code>SyntheticBanditDataset</code>，自定义 context-dependent 奖励函数和行为策略，在 5000 条日志、固定种子 42 下运行 DM / IPS / SNIPS / DR，并把同一个 <code>q̂</code> 输入 OBP 0.4.1 官方估计器做数值对照。

~~~bash
conda run -n openbandit python ope_demo.py
conda run -n openbandit python -m pytest -q
~~~

- 理论笔记：[docs/ope_theory.md](docs/ope_theory.md)
- 运行结果与验收：[docs/week1_results.md](docs/week1_results.md)
- 原始结果：[results/week1/ope_demo_results.csv](results/week1/ope_demo_results.csv)
- 汇总结果：[results/week1/ope_demo_summary.json](results/week1/ope_demo_summary.json)
- 测试：20 passed；本地实现与 OBP 官方实现的最大绝对差 <code>1.11e-16</code>。

核心接口：

~~~python
from ope.reward_model import cross_fit
from ope.estimators import evaluate

q_hat = cross_fit(feedback, folds=3, seed=42)
values = evaluate(feedback, action_dist, q_hat)
~~~

- <code>feedback</code>：与 OBP <code>bandit_feedback</code> 一致的字典，至少含 <code>context</code>、<code>action</code>、<code>position</code>、<code>reward</code>、<code>pscore</code>、<code>n_actions</code>、<code>len_list</code>。
- <code>action_dist</code>：<code>(n, n_actions, len_list)</code>，也可传 <code>(n, n_actions)</code> 表示位置无关策略；每行对动作求和为 1。
- <code>q_hat</code>：<code>(n, n_actions, len_list)</code>，由 <code>cross_fit</code> 三折交叉拟合得到。
- <code>evaluate</code> 返回 <code>DM / IPS / SNIPS / DR</code> 四个标量；可选 <code>clip</code> 限制重要性权重。

没有 OBP 的机器可使用本地回退：<code>python ope_demo.py --dataset local --no-save</code>。在项目根目录复现 D 的真实日志样例评估（先运行 `python -m scripts.prepare_data --sample`）：

```bash
python -m scripts.train_policies --processed-dir data/processed_sample --out-dir outputs/policies/sample
python -m scripts.evaluate_mse --processed-dir data/processed_sample --policy-dir outputs/policies/sample --out-dir results/ope_mse/sample
```

已有 C 的策略导出时直接运行第二条。全量数据使用默认路径，运行 `python -m scripts.evaluate_mse`。每个 campaign 默认从 BTS/test 独立抽取 20 组各 2,000 行，组内无放回、组间可重叠。每组重新三折拟合奖励模型，两种策略共享抽样及预测。以相应策略在 Random/test 的 IPS 估计为固定参考值，逐次平方误差写入 `replicates.csv`，按策略和估计器取平均得到 [MSE 汇总](results/ope_mse/sample/summary.csv)。参考值自身也有抽样误差。权重全零时 SNIPS 无定义，汇总同时给出有效次数。

## 成员 E · 敏感度实验、图表与 Poster

实验入口复用 B 的日志加载、C 的冻结策略和 D 的四种估计器。同一子样本只交叉拟合一次奖励模型，全部目标策略、固定截断上限与正权重分位数规则共享预测。正式配置保存在 config.yaml 的 experiments 段。

已附正式结果。仅重绘图表、更新报告或海报素材时，在项目根目录运行以下命令。

~~~powershell
uv sync --locked
uv run python -m scripts.plot_sensitivity --poster-assets-dir poster --report docs/member_e_experiments.md
~~~

从头复算时，先准备处理后的日志和冻结策略。已具备这两项输入时，直接从 `run_sensitivity` 开始。下例使用新的输出目录，以便与正式交付逐项比较。

~~~powershell
uv run python -m scripts.prepare_data
uv run python -m scripts.train_policies --out-dir outputs/policies/e_full
uv run python -m scripts.run_sensitivity --out-dir tmp/e_reproduction
uv run python -m scripts.plot_sensitivity --run-dir tmp/e_reproduction --out-dir tmp/e_reproduction/figures
~~~

中断后对同一输出目录用 `--resume` 恢复，也可显式增加 `--repeats`。每个运行记录输入、源码、环境和 CSV 摘要。新目录会获得新的运行 ID。子样本和点估计使用固定实验种子，汇总表的 bootstrap 种子还使用运行 ID，因此新运行的区间端点可能略有差异。

官方样例可用以下命令检验数据流。样例的两种日志使用各自时间窗口。

~~~powershell
uv run python -m scripts.prepare_data --sample
uv run python -m scripts.train_policies --processed-dir data/processed_sample --out-dir outputs/policies/e_sample
uv run python -m scripts.run_sensitivity --processed-dir data/processed_sample --policy-dir outputs/policies/e_sample --out-dir results/sensitivity/e_sample --window native --sample-sizes 500 1000 2000 --repeats 5 --clip-grid 5 20 100 none
~~~

正式结果位于 results/sensitivity/e_full。[实验分析报告](docs/member_e_experiments.md)收录七张主图，包括三个 campaign 的截断对照、三张样本量图、方差柱状图、权重诊断和排序一致率。每张图在 results/figures/e_full 提供 PDF 和 PNG。需要逐 campaign 的完整截断分解和正权重分位数曲线时，在绘图命令后加 `--details-dir tmp/e_figure_details`。[最终 Poster](poster/poster.pdf)的数值可通过 poster/results_provenance.json 追溯到同一组实验结果。

~~~powershell
uv run pytest -q -rs
latexmk -pdf -cd poster/poster.tex
# 也可使用已安装的 Tectonic
tectonic --keep-logs poster/poster.tex
~~~

完整实现依据见[成员 E 实现方案](docs/member_e_implementation_plan.md)。

## 参考

Saito, Y., Aihara, S., Matsutani, M., & Narita, Y. (2020). *A Large-scale Open Dataset for Bandit Algorithms.* arXiv:2008.07146.
