# 成员 E 后续工作详细实现方案

本方案已实施。全量来源日志的 750 组重复实验产生 84,000 条估计结果，实测分析与复现入口见[成员 E 实验报告](member_e_experiments.md)。以下内容保留方案形成时的审计与设计依据。

本方案对应 2026-11-17 的仓库状态，基准提交为 d5cf726d388cf170f9b340601b1cb0f4e0281215。交付范围按本轮确认聚焦于实现设计，包含 Git 历史审计、初步方案的反驳与修正、代码接口、实验协议、图表与 Poster 交付要求、验收步骤。

## 1. E 的职责与交付目标

E 负责权重截断敏感度实验、不同样本量下的偏差与方差分析、Matplotlib 图表资产，以及答辩 Poster 的实验结果更新。提案还要求策略排序稳定性，因此方案包含与现有两种目标策略匹配的排序统计。[分工表](../README.md#分工与进度) [提案第 2—3 页](../openbanditproposal.pdf)

后续实现应形成以下完整链路。

~~~text
B 的处理后日志 + C 的冻结策略分布
                    ↓
输入核验、共同测试时间窗口、Random 日志参考值
                    ↓
BTS 重复子抽样 + D 的 cross_fit / evaluate
                    ↓
截断扫描、样本量扫描、权重诊断、排序统计
                    ↓
逐次结果 CSV → 汇总 CSV → Matplotlib 图表
                    ↓
E 的实验分析报告 + 更新后的 A4 Poster
~~~

课程文件将 Poster 截止时间列为 2026-11-10 18:00，报告、Slides 和录制展示截止时间列为 2026-11-24 18:00。11 月 9 日的海报提交符合前期展示阶段。后续工作优先产出可供最终报告与展示复用的实验图表，再将实测结论整合进现有海报。报告正文上限为 15 页，E 的分析宜整理为可供 A 合并的独立小节。[课程要求第 1—3 页](../final_project_des.pdf)

## 2. Git 历史审计

### 2.1 历史范围与完整性

本地仓库为完整克隆，所有现有引用可达的提交共 9 个。通过 GitHub 接口核对，远程 main 与 feat/fengjingzhe-OPE 均指向 d5cf726，远程标签 v0.1-data-pipeline 与本地一致。分析覆盖全部现有分支和标签的可达历史。[远程分支](https://github.com/L-16643/6007/branches) [提交历史](https://github.com/L-16643/6007/commits/main/)

| 日期，UTC+8 | 提交 | 实际内容 | 对 E 的意义 |
|---|---|---|---|
| 11-07 22:42 | ada01de | 远程初始化，仅两行 README | 项目起点 |
| 11-08 01:38 | 39d6ca3 | B 的完整数据流水线、特征工程、参考值计算、配置、测试与结果 | E 的日志格式、位置索引和参考值接口均已建立 |
| 11-08 01:41 | 1c4a564 | 合并初始化历史和 B 的历史，树内容与 B 父提交一致 | 两个根提交接入同一历史，标签标记数据流水线版本 |
| 11-08 14:19 | 91f0e1b | 加入课程要求和项目提案 PDF | 明确 E 的样本量实验、Matplotlib 图表和 A4 Poster 交付 |
| 11-09 16:46 | 2377657 | 加入 300 行 LaTeX 海报及 PDF | 已有可延续的英文三栏版式 |
| 11-10 20:03 | 2e85371 | C 的 LinUCB、LogisticTS、导出脚本、三份全量清单和测试 | 提供 Random/test 与 BTS/test 上分别对齐的策略分布 |
| 11-10 20:10 | 8609af3 | README 将 C 标记为完成 | 状态更新，算法来自前一提交 |
| 11-13 00:33 | 0e75b23 | D 在数据流水线分支上加入四种估计器、cross-fitting、合成实验和测试 | 截断参数从这一版起已经存在 |
| 11-17 16:26 | d5cf726 | 合并 D 分支与 C 所在主线，同时新增真实日志重复子抽样 MSE 脚本和结果 | 当前集成基线，补充了零权重时 SNIPS 的处理 |

提交关系如下。

~~~text
ada01de ──────────┐
                 ├─ 1c4a564 ─┬─ 91f0e1b ─ 2377657 ─ 2e85371 ─ 8609af3 ─┐
39d6ca3 ─────────┘           └─ 0e75b23 ────────────────────────────────┴─ d5cf726
~~~

### 2.2 从历史得到的实施依据

- B 的数据模块、数据入口、配置和依赖文件从 39d6ca3 到当前提交内容一致，适合直接复用。
- C 的策略模块与导出脚本从 2e85371 到当前提交内容一致，冻结策略及其行序清单已形成明确接口。
- 海报从 2377657 到当前提交内容一致，仍包含 planned、初步 CTR 结果和后续实验展望。E 的工作需要将实验部分更新为实测内容。
- d5cf726 是包含新功能的合并提交。它加入了 88 行 MSE 脚本、480 条重复实验记录、24 条汇总记录，以及零权重测试。
- D 的文档保留 Week 1 阶段描述，当前实现能力应结合最新代码与合并提交判断。其估计器已经具备 E 所需的 clip 参数。
- C 的全量清单已经入库，大型概率数组由 outputs/.gitignore 排除。当前工作区只有清单与历史结果，真实日志和策略数组需要在实现阶段准备。

对应代码锚点为数据加载器、C 的导出清单、D 的评估函数与重复实验入口。[数据加载器](../ope/data/dataset.py) [策略导出](../scripts/train_policies.py) [估计器](../ope/estimators.py) [MSE 脚本](../scripts/evaluate_mse.py)

## 3. 本轮实际验证结果

### 3.1 测试基线

执行了以下命令。

~~~powershell
python -m pytest -q -rs -p no:cacheprovider
~~~

结果为 **22 passed，4 skipped**。

| 跳过项 | 原因 |
|---|---|
| 两项真实样例数据测试 | 当前工作区尚无官方样例 |
| 一项 OBP 估计器对照测试 | 当前解释器未安装 OBP |
| 一项 OBP 合成数据接口测试 | 当前解释器未安装 OBP |

本轮使用现有 Python 3.13.5 环境，NumPy 2.1.3、pandas 2.2.3、SciPy 1.15.3、scikit-learn 1.6.1、PyArrow 19.0.0、pytest 8.3.4。该环境与项目锁文件存在版本差异。后续正式复现实验需要完成锁定环境安装，并记录实际依赖版本。

### 3.2 历史结果复核

对已提交的两个样例 CSV 做了重算。

- 480 条逐次结果按 campaign、policy、estimator 汇总后，与原 MSE 表的最大绝对差为 5.29 × 10⁻¹⁷。
- 使用分母 R 的经验方差，MSE 与“经验偏差平方加经验方差”的最大分解残差为 4.34 × 10⁻¹⁹。
- LinUCB 的 60 次 IPS 估计全部为零。
- Men、Women 的 LinUCB 参考值也为零，因而对应 IPS 的样例 MSE 为零。
- 共有 56 条估计值为负，均出现在 DR。绘图和汇总需要保留这些原始估计。
- All 的样例 LinUCB 参考值为 0.0266667，C 的全量清单参考值约为 0.00562765。两者对应不同数据规模及训练产物，结果表需要携带 source 和输入身份。

这些事实决定了样例运行的验收重点应为数据流与算术正确性，正式结论应来自全量来源日志上的实验，并同时展示参考值与覆盖诊断。[逐次结果](../results/ope_mse/sample/replicates.csv) [MSE 汇总](../results/ope_mse/sample/summary.csv) [全量策略清单](../results/policies/all/manifest.json)

## 4. 初步方案及自我反驳

初步方案是沿 D 的重复子抽样方法增加样本量与截断阈值循环，调用现有四种估计器，生成 MSE、偏差、方差和 ESS 图表，再填入现有海报。

这个方案的复用方向合理，以下细节需要修正。

| 初步设想 | 反驳依据 | 修正后的设计 |
|---|---|---|
| 直接读取 C 清单中的参考值 | 清单中的参考值有抽样误差，并绑定当时完整的 Random/test | 从当前输入和实际评估时间窗口重算，保留标准误、区间和参考贡献数 |
| 沿用两种日志各自的整个 test | 两种日志独立按行数切分，起始时间存在差异 | 正式实验使用共同测试时间窗口，分布可比性作为明确解释条件 |
| 每个阈值独立抽样、拟合 q | 阈值效应会混入抽样与拟合噪声，训练成本也随阈值数放大 | 同一 campaign、n、repeat 共享样本和 q，随后扫描所有策略与阈值 |
| 全量计算一次 q 后对子样本切片 | 奖励模型训练规模和拟合随机性发生变化，样本量实验的统计对象随之改变 | 每个重复子样本重新 cross-fit，阈值循环复用这次预测 |
| 用 pscore 下限实现截断 | pscore 下限产生的权重上限取决于目标概率，和 D 的固定权重上限含义不同 | 主实验采用 D 已实现的 min(w, c) |
| 对全部权重直接取高分位数 | 确定性策略可能有大量零权重，高分位点仍可能为零 | 分位数扩展明确使用正权重分位点，并保存实际阈值及零权重比例 |
| 用默认 pandas 方差完成分解 | pandas 默认分母为 R−1，和 MSE 的分母 R 不匹配 | 分解使用 ddof=0，Monte Carlo 标准误另用 ddof=1 |
| 对缺失的 SNIPS 直接填零或忽略 | 填零改变估计器，静默忽略会掩盖有效次数变化 | 保存状态、有效次数和无效比例，缺失估计保留为空 |
| 用样例 MSE 为零说明估计器优秀 | 当前样例出现参考值和估计值同时为零的退化 | 记录非零参考贡献数、点击数与覆盖状态，样例结论标明适用范围 |
| 认为截断后的 DR 自动保留完整双重稳健性 | 截断修改了残差校正权重 | 理论说明限定到实际估计公式，并通过数据观察偏差—方差变化 |
| 对两个目标策略直接报告平均 Spearman | 两策略无并列时相关系数仅取 +1 或 −1 | 同时报告排序一致率、并列率、有效次数和参考排序的不确定性 |
| 在扫描结果里选择最小 MSE 并宣称最优 | 同一批结果同时参与选择与评价，容易产生选择乐观偏差 | 全部网格公开展示，将最低点描述为本次网格观察结果 |
| 将多个阈值或多种估计器当作重复样本 | 它们共享同一批样本，存在配对相关 | 统计重采样单位统一为 repeat，比较阈值时按 repeat 配对 |
| 直接为全量 BTS 分配 q 张量 | All 的完整 float64 q 约需 7.12 GB，后续计算还会产生临时数组 | 策略分布 mmap，q 只为当前子样本分配，参考值按块计算 |

### 4.1 时间窗口差异的具体证据

现有全量清单记录的 test 起始时间如下，均为 UTC。

| Campaign | Random/test 起始 | BTS/test 起始 |
|---|---|---|
| All | 2019-11-28 17:04:50.826684 | 2019-11-28 16:54:52.468194 |
| Men | 2019-11-28 15:49:53.338691 | 2019-11-28 15:37:40.799547 |
| Women | 2019-11-28 15:14:51.326000 | 2019-11-28 15:23:48.035014 |

All 与 Men 的 BTS/test 早期区间还与 C 使用的 Random/train 时间段重叠。正式方案使用两种 test 的时间交集，并要求评估行晚于 Random/train 的最后时间戳。对分布数组使用相同的原始行索引，随后重新计算参考值。

共同时间窗口减少了这一已知差异。上下文分布、位置比例和时间相关性仍需要在报告中描述，参考误差口径可以涵盖剩余分布差异。

## 5. 修正后的统计对象与数学定义

### 5.1 评估对象

每条记录对应一个推荐位曝光。对 campaign 固定 C 导出的 LinUCB 与 LogisticTS，两种策略都使用已导出的概率数组。

主要输出是给定日志、冻结目标策略和重复子抽样机制下，相对于固定 Random 日志参考值的经验误差与波动。

采用以下符号。

- pᵢ 为原始日志 propensity score。
- πᵢ 为目标策略对已展示动作、已观察位置的概率。
- wᵢ = πᵢ / pᵢ。
- wᵢ⁽ᶜ⁾ = min(wᵢ, c)，无截断时使用 wᵢ。
- qᵢ 为 cross-fitting 给出的已展示动作奖励预测。
- mᵢ 为当前观察位置上对目标动作分布加权的奖励预测。

四个估计值沿用 D 的计算。

\[
\widehat V_{\mathrm{DM}}=\frac1n\sum_i m_i
\]

\[
\widehat V_{\mathrm{IPS},c}=\frac1n\sum_i w_i^{(c)}r_i
\]

\[
\widehat V_{\mathrm{SNIPS},c}
=\frac{\sum_i w_i^{(c)}r_i}{\sum_i w_i^{(c)}}
\]

\[
\widehat V_{\mathrm{DR},c}
=\frac1n\sum_i\left[m_i+w_i^{(c)}(r_i-q_i)\right]
\]

SNIPS 的分母为零时结果无定义，使用 evaluate 的 allow_zero_mass=True 返回 NaN。此时 DM、IPS 和 DR 继续保留其定义下的结果。无截断使用 Python None，传入正无穷会触发现有接口的参数校验。

这些公式对应位置条件下的边际策略价值。推广到整页联合推荐效果需要额外的联合策略与位置交互假设。

### 5.2 固定参考值

在选定的 Random/test 行上构造

\[
z_j=\frac{\pi_e(a_j\mid x_j,\ell_j)}{p_j}r_j,\qquad
g=\frac1M\sum_j z_j.
\]

校验 pⱼ = 1/K。参考值同时保存以下信息。

- 参考窗口、样本数、策略名与 source。
- value、se、normal_ci_low/high。
- bootstrap_ci_low/high，以及区间方法。
- mean_weight、max_weight、ESS。
- positive_weight_count、positive_contribution_count。
- 出现全零贡献时的 reference_status。

先按块核验策略分布并收集一维贡献向量 z，再复用 B 的 bootstrap_mean_ci。标准误使用 std(z, ddof=1)/sqrt(M)。通过小数据测试与 B 的 policy_value_from_random_logs 逐项对照，保证算术口径一致。

参考贡献全部为零时，经验 bootstrap 区间会退化。保存原始结果与退化状态，图注说明该参考的低信息量。相对误差只有在参考值严格为正且可解释时才有定义，主结果采用绝对量纲。

OBP 的标准基准利用评估策略自身日志上的观测回报作为参考。当前项目对新训练策略使用 Random 日志 IPS 参考，需要在方法部分写明这一项目设定。[OBP 评估协议](https://zr-obp.readthedocs.io/en/latest/evaluation_ope.html)

### 5.3 偏差、方差与 MSE

对同一 campaign、策略、样本量、阈值和估计器，令有限估计的集合为 I，有效次数为 Rᵥ，固定参考值为 g。

\[
\bar V=\frac1{R_v}\sum_{b\in I}\widehat V_b
\]

\[
\mathrm{bias}_{ref}=\bar V-g
\]

\[
\mathrm{variance}_{emp}
=\frac1{R_v}\sum_{b\in I}(\widehat V_b-\bar V)^2
\]

\[
\mathrm{MSE}_{ref}
=\frac1{R_v}\sum_{b\in I}(\widehat V_b-g)^2
=\mathrm{bias}_{ref}^2+\mathrm{variance}_{emp}
\]

经验分解对于固定参考值和相同有效重复集合成立。代码字段明确采用 bias_ref、variance_emp、mse_ref，报告中文写作“相对参考值的经验偏差”“重复子抽样经验方差”“参考 MSE”。

若另用样本方差 s²，分母为 Rᵥ−1，则恒等式中的方差项为 (Rᵥ−1)s²/Rᵥ。

真实策略值 V 通常未知。若参考 g 和估计值独立、参考对同一个 V 无偏，才有

\[
\mathbb E[(\widehat V-g)^2]
=\mathbb E[(\widehat V-V)^2]+\mathrm{Var}(g).
\]

因此结果页同时报告参考的不确定性。主结果保留原始 mse_ref，参考误差方差的直接扣减需要额外统计假设。

### 5.4 截断的理论解释

在正确倾向得分、可比上下文分布及相应条件独立性下，IPS 截断的期望偏移由

\[
\mathbb E[(w^{(c)}-w)q]
\]

控制。二元奖励的条件均值非负，因此该项非正。

DR 的截断偏移包含

\[
\mathbb E[(w^{(c)}-w)(q-\widehat q)].
\]

其符号取决于奖励模型误差与权重的关系。真实条件奖励模型成立时，加权残差的条件期望仍为零。报告应围绕实测 bias_ref、variance_emp 和 mse_ref 展开，截断导致的偏差—方差变化有直接相关的 OPE 研究依据。[Su 等，ICML 2020](https://proceedings.mlr.press/v119/su20a.html)

### 5.5 统计区间与配对比较

- 每组报告 repeats、valid_repeats、invalid_rate。
- Rᵥ ≥ 2 时，均值的 Monte Carlo 标准误为 std(estimate, ddof=1)/sqrt(Rᵥ)。
- MSE 的 Monte Carlo 标准误可由逐次 squared_error 的样本标准差除以 sqrt(Rᵥ) 得到。
- 单组误差区间在有效 repeat 上重采样生成，并注明条件于当前日志、当前参考值及估计值有定义。无效比例单独报告。
- 比较阈值 c 与无截断结果时，对同一 repeat 的 squared_error 做差，再汇总配对差。
- 相关比较使用共同有效 repeat，额外报告 paired_valid_repeats。
- 全部估计无效时，各误差统计为空。仅一次有效时保留代数分解值，Monte Carlo 标准误和区间标记为信息不足。

跨记录时间相关性和共同用户相关性影响总体推断。当前实验定义为固定日志上的重复子抽样评测，其区间解释限定在该实验机制下。

## 6. 实验协议

### 6.1 主实验参数

以下为方案推荐的起始配置，正式运行前用小规模计时验证计算成本。

| 参数 | 正式实验推荐值 |
|---|---|
| campaign | all、men、women |
| 目标策略 | linucb、logisticts |
| 估计器 | DM、IPS、SNIPS、DR |
| 日志来源 | 全量来源的 processed 日志 |
| 时间窗口 | 两种 test 的交集，且晚于 Random/train 结束 |
| 样本量 | 1,000、2,000、5,000、10,000、20,000 |
| 重复次数 | 50，支持显式增加 |
| 固定权重上限 | 1、5、10、20、50、100、500、1,000、None |
| 拟合方式 | 每个子样本重新进行 3 折 cross-fitting |
| 主种子 | 42 |
| 抽样方式 | 每组内无放回，不同组独立抽取 |
| 参考 bootstrap 次数 | 1,000 |
| repeat 层指标 bootstrap 次数 | 2,000 |

上述固定网格对应 750 次 cross_fit 和 54,000 条估计器结果。每次拟合的预测供两种目标策略和全部阈值使用。同时启用下述五个分位点时，总计 84,000 条估计器结果，cross_fit 次数仍为 750。

50 次重复是起始计算配置。报告同时给出 Monte Carlo 不确定性，用实际稳定性评估是否增加重复次数。新增重复按原编号继续生成。

### 6.2 样例协议

样例采用 n = 500、1,000、2,000，重复 5 次，截断值为 5、20、100、None。

官方样例的两种日志可能覆盖不同时间范围。冒烟测试显式使用 window=native，并记录各自窗口。正式统计使用 window=intersection。样本量必须小于选定 BTS 候选行数，超限时显示实际可用行数并要求调整参数。

### 6.3 分位数截断扩展

提案明确提到分位数截断，正式运行可同时生成固定上限与分位数两组结果，共享同一次子样本拟合。分析时先解释固定上限主结果，再解释自适应分位数规则。

\[
c_{b,q}=Q_q\left(\{w_i:w_i>0\}\right)
\]

推荐 q 为 0.90、0.95、0.99、0.995、0.999，使用显式固定的 NumPy quantile method，例如 linear。每个 repeat、每个策略记录实际 c 值。

- 分位数只依赖该次样本的权重。
- 同一策略的所有估计器共享相同分位数阈值。
- 分位数阈值属于自适应截断规则，结果按 q 分组。
- 正权重集合为空时，threshold_status 标记为 zero_mass。此时用原始零权重计算仍有定义的估计器，SNIPS 保留缺失。
- 主实验和分位数实验使用一致的样本种子及拟合种子，便于配对核验。

### 6.4 抽样、随机种子与拟合

固定 campaign 编码为 all=0、men=1、women=2。使用 NumPy SeedSequence 从以下整数派生随机数。

~~~text
[master_seed, campaign_code, n, repeat, stream_id]
~~~

stream_id 为 0 时控制抽样，为 1 时生成 cross_fit 使用的 32 位整数种子。由此保证重排阈值列表、重排 campaign、单独重跑某个 n 或追加 repeat 时，已有单元的样本与预测可重现。

抽样后保留原始行索引。所有反馈向量、上下文和策略分布使用同一索引切片。原始行索引的 SHA-256 写入逐次结果，必要时可结合种子重建。

分层或时间块抽样会定义新的实验机制，主实验明确使用与 D 相衔接的均匀行子抽样。记录 sample_fraction=n/N。对于固定逐行分数的线性估计，无放回抽样具有有限总体修正，其形式为 (1−n/N)S²/n。DM、DR 的重拟合会引入额外随机性，报告保留实际测得的经验方差。

### 6.5 权重诊断

对原始权重和截断权重分别计算。

\[
ESS=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}
\]

诊断字段包括 mean_weight、max_weight、positive_weight_fraction、clip_fraction、removed_weight_fraction、ess_raw、ess_clipped、ess_fraction、n_clicks。

其中 clip_fraction = mean(w > c)，removed_weight_fraction = sum(w−w_clipped)/sum(w)。零权重时将 ESS 的诊断值约定为 0，同时记录 zero_mass 状态，分母相关的比例保存为空。

ESS 衡量权重集中程度，结合目标策略概率和奖励残差解读。它可用于解释曲线，单独的 ESS 大小无法确定参考 MSE 的排序。

### 6.6 排序稳定性

当前只有两种候选策略。每个 campaign、n、clip、estimator、repeat 计算估计差

\[
\widehat\Delta_b
=\widehat V_b(\mathrm{LinUCB})-\widehat V_b(\mathrm{LogisticTS}).
\]

参考差为 Δref = gLinUCB − gLogisticTS。参考差区间使用同一批 Random 行的配对贡献差 zLinUCB − zLogisticTS，再调用 B 的 bootstrap_mean_ci。

输出以下指标。

- 与经验参考顺序一致的比例。
- 两策略估计并列的比例，绝对容差固定为 10⁻¹² 并写入清单。
- paired_valid_repeats。
- 非并列重复上的 Spearman 均值。
- reference_rank_status，参考差区间跨零时标记 uncertain。

两策略均严格有序时 Spearman 只有 +1 和 −1，此时平均 Spearman 等于 2 × 排序一致率 − 1。存在并列时单独统计其比例。参考排序不确定的结果仍可展示经验一致率，并在图注中注明参考顺序的统计状态。

## 7. 文件级实现方案

### 7.1 目录与职责

~~~text
ope/experiments/
    __init__.py
    inputs.py
    sensitivity.py
    plotting.py

scripts/
    run_sensitivity.py
    plot_sensitivity.py

tests/
    test_sensitivity.py
    test_sensitivity_cli.py

docs/
    member_e_experiments.md

results/sensitivity/<run>/
    manifest.json
    references.csv
    replicates.csv
    diagnostics.csv
    summary.csv
    ranking.csv
    checkpoints/

results/figures/<run>/
    *.pdf
    *.png

poster/
    poster.tex
    poster.pdf
    generated_results.tex
~~~

新增 Matplotlib 依赖并更新 uv.lock。在 config.yaml 增加 experiments 配置段，由 E 的 CLI 读取，命令行显式参数覆盖配置值。README 的 E 章节补充复现入口、产物位置和已完成的验收范围。

### 7.2 inputs.py

建议提供以下直接接口。

~~~python
def load_inputs(campaign, processed_dir, policy_dir):
    """Load feedback, policy memmaps, metadata, and export manifest."""

def select_test_rows(inputs, window):
    """Return original Random/BTS row indices and effective time bounds."""

def compute_references(inputs, random_rows, *, bootstrap_repeats, seed):
    """Compute per-policy references and the paired reference difference."""
~~~

输入核验要求覆盖以下实际衔接点。

1. campaign、source、data_splits 与 context_feature_names 对齐。
2. Random 与 BTS 的完整行键摘要均按 C 的格式重算。
3. 摘要使用 timestamp、action、position 三列，按行、小端 64 位整数拼接。分块更新 SHA-256，保证与 C 的整体计算等价。
4. 策略分布形状同时与清单和实际反馈维度一致，概率有限、非负、沿动作轴归一化。
5. 验证 action、position 的范围，pscore 位于 (0,1]，reward 为二元值。
6. Random 日志 pscore 与 1/K 一致。
7. 对齐完整导出后再选择时间窗口，策略分布切片始终使用原始行号。
8. 时间戳在已处理数组中按序排列。旧版 meta 缺少时间戳单位时，将 s、ms、us、ns 作为候选单位，与 meta 的 UTC 首尾时间逐一核对，唯一匹配后统一为纳秒比较。读取 Random/train 的最后时间戳也执行同样核验。无法确定单位时给出输入错误。行键摘要始终使用导出的原始整数。
9. 记录输入文件和配置的摘要，支持重跑核对。
10. 旧版 C 清单包含行键摘要和特征名，其证据范围写入 provenance_status。本次文件摘要用于标识本次输入，策略生成时的特征文件身份由历史清单可提供的信息限定。

参考计算按块收集一维权重和贡献。对 2D 概率分布沿用 B/D 的位置无关语义，对 3D 分布按实际 position 索引。

### 7.3 sensitivity.py

建议将实验逻辑与 CLI 解耦。

~~~python
def run_replicate(
    inputs,
    bts_rows,
    *,
    n,
    repeat,
    master_seed,
    folds,
    clip_grid,
    quantile_grid,
):
    """Return estimator records and weight diagnostics for one paired run."""

def summarize_estimates(estimates, reference):
    """Compute valid counts, empirical bias, variance, MSE, and MC errors."""

def summarize_rankings(records, references):
    """Summarize paired policy ordering with tie and reference-status metadata."""
~~~

核心执行结构如下。这里展示参数与执行顺序，记录构造和输入验证由对应函数补齐。

~~~python
sample_seed = np.random.SeedSequence(
    [master_seed, campaign_code, n, repeat, 0]
)
q_seed = np.random.SeedSequence(
    [master_seed, campaign_code, n, repeat, 1]
)
rng = np.random.default_rng(sample_seed)
indices = rng.choice(bts_rows, size=n, replace=False)

sample = {
    key: feedback[key][indices]
    for key in ("context", "action", "position", "reward", "pscore")
}
sample.update(
    n_rounds=n,
    n_actions=feedback["n_actions"],
    len_list=feedback["len_list"],
)

fit_seed = int(q_seed.generate_state(1, dtype=np.uint32)[0])
q_hat = cross_fit(sample, folds=folds, seed=fit_seed)

for policy_name, distribution in policy_distributions.items():
    action_dist = distribution[indices]
    thresholds = resolve_thresholds(
        sample, action_dist, clip_grid, quantile_grid
    )
    for threshold in thresholds:
        values = evaluate(
            sample,
            action_dist,
            q_hat,
            clip=threshold.value,
            allow_zero_mass=True,
        )
        # Store one record per estimator and one set of weight diagnostics.
~~~

在同一 repeat 中，DM 在不同阈值下应相等。绘图时按相应统计分组显示一次，重复次数从 repeat 编号计算。

完整的统计核心可以按以下形式实现。

~~~python
def summarize_estimates(estimates, reference):
    values = np.asarray(estimates, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Expected a nonempty one-dimensional estimate array")
    if not np.isscalar(reference) or not np.isfinite(reference):
        raise ValueError("Expected a finite scalar reference")
    valid = values[np.isfinite(values)]
    result = {
        "repeats": len(values),
        "valid_repeats": len(valid),
        "invalid_rate": float(1 - len(valid) / len(values)),
    }

    if len(valid) == 0:
        result.update(
            mean_estimate=np.nan,
            bias_ref=np.nan,
            variance_emp=np.nan,
            mse_ref=np.nan,
            rmse_ref=np.nan,
            mean_mc_se=np.nan,
            mse_mc_se=np.nan,
        )
        return result

    mean_estimate = float(valid.mean())
    errors = valid - reference
    squared_errors = errors ** 2
    bias = mean_estimate - reference
    variance = float(np.mean((valid - mean_estimate) ** 2))
    mse = float(squared_errors.mean())

    result.update(
        mean_estimate=mean_estimate,
        bias_ref=bias,
        variance_emp=variance,
        mse_ref=mse,
        rmse_ref=float(np.sqrt(mse)),
        mean_mc_se=np.nan,
        mse_mc_se=np.nan,
    )

    if len(valid) >= 2:
        result["mean_mc_se"] = float(valid.std(ddof=1) / np.sqrt(len(valid)))
        result["mse_mc_se"] = float(
            squared_errors.std(ddof=1) / np.sqrt(len(valid))
        )

    if not np.isclose(mse, bias ** 2 + variance, rtol=1e-10, atol=1e-14):
        raise ArithmeticError("Empirical MSE decomposition failed")
    return result
~~~

调用层按完整分组键传入记录，并检查重复主键。分解断言的容差作为明确参数记录在测试中。上述片段中的 NumPy import 由模块统一提供。

拟合产生的 ConvergenceWarning 需要记录计数及具体实验单元。拟合异常或非有限预测使该单元失败，并保留可定位的错误信息，后续恢复重新执行该单元。

### 7.4 run_sensitivity.py

CLI 负责配置解析、输入预检、进度显示、结果发布与恢复。

建议参数如下。

~~~text
--config
--campaign
--processed-dir
--policy-dir
--out-dir
--window intersection|native
--sample-sizes
--repeats
--folds
--clip-grid
--quantile-grid
--seed
--reference-bootstrap
--metric-bootstrap
--resume
~~~

参数校验覆盖非正或重复样本量、非法阈值、NaN/Inf、非法分位点、负种子、重复次数不足和 folds 大于样本量等情况。none 映射为 None，显式 clip 值必须有限且大于零。

### 7.5 plotting.py 与 plot_sensitivity.py

绘图入口读取已完成运行的 manifest 和 CSV。先检查 schema_version、分组唯一性、重复次数、有效性状态与 source 一致性，再生成静态资产。参数包括 --run-dir、--out-dir，以及可选的 --poster-assets-dir。最后一个参数显式选择需要更新的海报资产目录。

Matplotlib 使用 Agg 后端，PNG 以 300 DPI 输出，PDF 使用矢量曲线和嵌入字体。颜色和线型统一，例如 DM 蓝、IPS 橙、SNIPS 绿、DR 紫。字体与现有海报的 Times 风格衔接，字体回退情况写入生成记录。

## 8. 结果表结构与可恢复执行

### 8.1 表结构

| 文件 | 关键字段 |
|---|---|
| manifest.json | schema_version、run_id、Git 提交、源码摘要、输入摘要、source、配置、版本、抽样规则、时间窗口、完成状态 |
| references.csv | campaign、policy、window_id、value、se、区间、M、ESS、正权重数、非零贡献数、reference_status |
| replicates.csv | run_id、campaign、source、window_id、policy、estimator、n、repeat、sample_digest、fit_seed、clip_id、clip_mode、clip_value、quantile、reference、estimate、squared_error、status |
| diagnostics.csv | campaign、policy、n、repeat、clip_id、样本比例、点击数、原始与截断权重诊断、拟合告警 |
| summary.csv | 完整分组键、总次数、有效次数、无效比例、mean_estimate、bias_ref、variance_emp、mse_ref、rmse_ref、MC 标准误及区间 |
| ranking.csv | campaign、n、clip_id、estimator、参考差及区间、有效配对数、并列率、一致率、Spearman、参考排序状态 |

完整分组键包含 run_id、source、window_id、campaign、policy、estimator、n、clip_id。clip_id 采用 none、cap_20、qpos_0.99 等稳定标识。分位数实验按规则标识分组，实际数值阈值保存在逐次记录。

CSV 使用 UTF-8 和原始浮点精度，缺失数值留空并配合 status。JSON 使用严格有限数值约束，缺失值转为 null。百分比转换只发生在显示层。

### 8.2 恢复与一致性

以 campaign、n、repeat 为检查点单元，同一单元包含全部策略、阈值和估计器的结果。

1. 先写入运行 manifest，固化输入、代码和实验参数身份。
2. 每个单元在同一文件系统的临时位置完整写出，再原子替换为检查点文件。
3. 每个检查点携带配置摘要、唯一主键与记录数，恢复时核验完整性。
4. --resume 只复用计算参数与输入摘要相符的完整单元。计算身份覆盖输入、源码、环境、窗口、种子、folds 和阈值规则。新增 n 或 repeat 只扩展待计算单元集合，已有单元按原编号复用。修改阈值规则使用独立运行目录。
5. 汇总由已核验的检查点重新生成，按主键去重检查。
6. 汇总和图表只将覆盖全部预定单元的运行标记为 complete。
7. 同一输出运行采用单写入者约束，第二个进程得到明确的占用信息。异常退出后的恢复需核对写入进程已结束，再接管检查点。

该结构支持中断后继续计算，也能避免不同配置结果混入同一汇总。

## 9. 图表与 Poster 方案

### 9.1 必需图表

| 图表 | 展示内容 | 解释重点 |
|---|---|---|
| clipping_mse | 每个 campaign × policy，在固定 n 下比较全部截断阈值的 mse_ref | 截断对整体参考误差的影响 |
| clipping_decomposition | 同一设置下 bias_ref²、variance_emp 与 mse_ref | 检查偏差—方差分解，显示各估计器差异 |
| sample_size_all/men/women | 每个目标策略随 n 变化的 bias_ref、variance_emp、mse_ref 折线 | 样本量效应和重拟合协议 |
| weight_diagnostics | ESS/n、截断比例、移除权重比例随阈值变化 | 将曲线变化与权重分布联系 |
| ranking_agreement | 排序一致率、并列率、有效配对数 | 两策略顺序的重复抽样稳定性 |

图表分别导出 PDF 与 PNG。用于海报的单面板从同一绘图函数生成，以便按实际印刷尺寸调整标签。

具体绘图规则如下。

- bias_ref 使用带零线的有符号坐标。
- MSE 与方差可按数据动态范围选择科学计数或对数刻度。精确零值应明确标注，保持源数据数值。
- 无截断显示为独立的 None/∞ 类别点，和有限阈值区分。
- 图例同时使用颜色和线型，便于灰度阅读。
- 每张图附 source、n、R、参考口径和区间含义。
- 单位统一。估计值或偏差显示为百分点时乘 100，MSE 或方差显示为百分点平方时乘 10,000。
- 缺失或参考退化的设置显示其状态和有效次数。
- 样本量主图固定为无截断设置，其他阈值可另生成配套图，避免在一张图中混入过多曲线。

### 9.2 现有 Poster 的具体改动

使用现有 A4 横向三栏布局作为基础，按以下内容更新。

| 区域 | 后续内容 |
|---|---|
| Background | 项目问题与目标的简洁介绍 |
| Dataset | 数据规模、冻结策略、共同评估窗口与单推荐位口径 |
| Estimators | 四个估计器及统一的权重截断定义 |
| Experimental design | 实际 n 网格、R、阈值、cross-fitting 与参考值计算 |
| Results | 两张核心实测图及一张简短结果表 |
| Conclusion | 与数据一致的发现、参考误差和覆盖局限、后续扩展方向 |

海报流程图中，将 q 的来源画为 BTS 子样本内部的交叉拟合，并明确 C 的目标策略由 Random/train 得到。

主图预先选择 All campaign 与最大样本量，展示两种目标策略的截断表现。样本量图可采用 All/LogisticTS，完整 campaign 结果放在报告与独立资产中。该选择依据最大规模 campaign 和版面容量，属于本方案的展示设计决定。

显式传入 --poster-assets-dir 后，plot_sensitivity.py 从选定运行结果生成 generated_results.tex，海报通过 input 引入表格数值和实验参数，通过 includegraphics 引入 PDF 图表。更新后的 PDF 需要检查页数、A4 横向尺寸、字体、裁切、图例和文本溢出，并逐页渲染验证。

### 9.3 分析报告

docs/member_e_experiments.md 应包含实验设置、参考值表、主要图表、数据支持的解释，以及复现命令。结论按三个问题组织。

1. 截断在哪些 campaign、目标策略与样本量下改善了参考 MSE，配对差与 MC 不确定性有多大。
2. 样本量增加时，经验偏差与方差如何变化，低 ESS 或稀疏点击如何影响结果。
3. 两目标策略的排序稳定性如何，参考差区间是否支持确定顺序。

阈值扫描保留全部结果。最低 MSE 点注明为该运行网格中的观察值。若后续需要部署阈值推荐，应增加独立选择与评价数据的设计。

## 10. 后续运行命令

以下命令对应待实现的入口，便于实施后逐步验收。

### 10.1 环境与样例数据

~~~powershell
uv add matplotlib
uv sync --locked
uv run python -m scripts.prepare_data --sample
uv run python -m scripts.train_policies --processed-dir data/processed_sample --out-dir outputs/policies/e_sample
~~~

C 的训练脚本要求目标 campaign 输出目录为新目录。重新生成时使用新的 out-dir，清单和概率数组成套使用。

### 10.2 E 的冒烟运行

~~~powershell
uv run python -m scripts.run_sensitivity --processed-dir data/processed_sample --policy-dir outputs/policies/e_sample --out-dir results/sensitivity/e_sample --window native --sample-sizes 500 1000 2000 --repeats 5 --clip-grid 5 20 100 none
uv run python -m scripts.plot_sensitivity --run-dir results/sensitivity/e_sample --out-dir results/figures/e_sample
uv run pytest -q -rs
~~~

### 10.3 正式数据与计时

已有 B/C 的成套全量产物时，可直接将其路径传给 E。需要本机生成时使用现有入口。

~~~powershell
uv run python -m scripts.prepare_data
uv run python -m scripts.train_policies --out-dir outputs/policies/e_full
uv run python -m scripts.run_sensitivity --campaign men --policy-dir outputs/policies/e_full --out-dir results/sensitivity/e_pilot --window intersection --sample-sizes 2000 20000 --repeats 3 --clip-grid 20 none
~~~

记录两种规模的单次拟合耗时、峰值内存和预检耗时，据此估算正式网格成本。

### 10.4 正式实验与恢复

~~~powershell
uv run python -m scripts.run_sensitivity --policy-dir outputs/policies/e_full --out-dir results/sensitivity/e_full --window intersection --sample-sizes 1000 2000 5000 10000 20000 --repeats 50 --clip-grid 1 5 10 20 50 100 500 1000 none --quantile-grid 0.90 0.95 0.99 0.995 0.999
uv run python -m scripts.run_sensitivity --policy-dir outputs/policies/e_full --out-dir results/sensitivity/e_full --window intersection --sample-sizes 1000 2000 5000 10000 20000 --repeats 50 --clip-grid 1 5 10 20 50 100 500 1000 none --quantile-grid 0.90 0.95 0.99 0.995 0.999 --resume
uv run python -m scripts.plot_sensitivity --run-dir results/sensitivity/e_full --out-dir results/figures/e_full --poster-assets-dir poster
~~~

上述正式命令一次完成固定阈值与分位数实验。若固定阈值实验已经单独完成，可用下面的独立目录命令补充分位数实验。该补充运行按相同种子重新拟合，每个重复单元仍只拟合一次。

~~~powershell
uv run python -m scripts.run_sensitivity --policy-dir outputs/policies/e_full --out-dir results/sensitivity/e_quantile --window intersection --sample-sizes 1000 2000 5000 10000 20000 --repeats 50 --clip-grid none --quantile-grid 0.90 0.95 0.99 0.995 0.999
~~~

### 10.5 海报

~~~powershell
latexmk -pdf -cd poster/poster.tex
~~~

当前 PATH 中未发现 pdflatex 或 xelatex，后续海报构建需要可用的 LaTeX 环境。可使用现有团队编译环境或安装完整 TeX 工具链，记录实际构建方式。

## 11. 针对性验收测试

新增测试覆盖现有测试尚未验证、且直接关系 E 结论的情况。

| 测试 | 预期断言 |
|---|---|
| 原估计器无截断一致性 | 对相同 sample、q、π，E 调用结果逐项等于 D 的 evaluate |
| 截断手算 | 使用现有两行夹具，c=0.75 时 DM=0.4、IPS=0.375、SNIPS=0.6、DR=0.525 |
| DM 阈值不变性 | 同一 repeat 全部阈值下 DM 相同 |
| 极大阈值 | c 至少等于样本最大权重时，结果与 None 一致 |
| 零重叠 | SNIPS 缺失、IPS=0、DR=DM，状态和有效次数正确 |
| 有符号 DR | 负估计保留，CSV 及图表数据未被数值裁剪 |
| 分解恒等式 | [0.1,0.2,0.3] 相对 g=0.15 时，bias=0.05，variance=0.0066667，MSE=0.0091667 |
| 方差分母 | ddof=0 满足直接分解，ddof=1 经 (R−1)/R 修正后相符 |
| 缺失统计 | 全无效、一次有效、混合有效的计数和区间状态正确 |
| 共同时间窗口 | 行索引与概率数组同步过滤，评估行晚于策略训练结束 |
| 参考值重算 | 变更 reward 或窗口后参考值更新，按块结果与 B 接口一致 |
| 行序校验 | 任一日志或概率形状与清单不一致时返回明确错误 |
| 2D/3D 输入 | 位置无关广播与显式三维分布结果相同 |
| 分位数截断 | 大量零权重、全零权重、重复分位点时规则和状态正确 |
| 调用次数 | 每个 campaign、n、repeat 恰好调用一次 cross_fit |
| 随机数稳定性 | 调整阈值顺序、单独重跑 campaign、追加重复时已有结果一致 |
| 排序统计 | 顺序相同、相反、并列和缺失四种情况正确，参考差按行配对 |
| 恢复执行 | 人为中断后恢复与一次运行的逐次结果一致，主键唯一 |
| CLI 端到端 | 在小型生成数据上完成输入→CSV→图表→完成状态的链路 |

实际数据验收按以下顺序进行。

1. 在锁定环境通过自动测试。
2. 准备官方样例，实际运行 B、C、E 的标准入口。
3. 完成全量来源的计时实验，并检查参考值及时间窗口。
4. 完成主实验，核验记录数、分组数、有效次数及分解残差。
5. 确认图表使用同一运行 CSV，数值单位正确。
6. 验证 Poster 的所有实验数字能回到对应汇总行。
7. 完成分位数扩展及报告解释。
8. 汇总实际通过、跳过和失败的测试，再更新 README 的 E 状态。

## 12. 实施顺序与最终交付条件

建议以以下顺序实现，每一步都有可检查产物。

| 阶段 | 实施内容 | 验收产物 |
|---|---|---|
| 1 | inputs、共同窗口、参考值与诊断 | 对齐日志、参考值表、输入清单 |
| 2 | 单次配对实验、固定阈值扫描、统计函数 | 手算与分解测试通过 |
| 3 | CLI、检查点、恢复、样例运行 | 完整样例 CSV 和恢复验证 |
| 4 | 正式计时与主实验 | 三个 campaign 的完整运行结果 |
| 5 | 分位数扩展、排序与不确定性 | 扩展结果和可解释的排序表 |
| 6 | Matplotlib 图表与分析报告 | PDF/PNG 资产和实验说明 |
| 7 | Poster 数值联动、编译和视觉验收 | 一页 A4 横向最终海报 |

以 11 月 24 日提交为目标，可将 11 月 18—19 日用于代码与样例验收，20—21 日用于正式实验，22 日完成图表和分析，23 日完成海报与复现检查。实际运行安排根据计时结果调整。

E 完整交付应满足以下条件。

- 固定阈值和样本量实验可从 CLI 复现。
- 各图表同时具有逐次 CSV、统计定义和输入身份。
- 相对参考值的经验偏差、经验方差及 MSE 分解一致。
- 稀疏点击、零权重、缺失估计和参考排序不确定性有显式状态。
- 实验分析报告覆盖实测发现、适用条件和局限。
- 最终 Poster 使用实际实验结果，并完成编译与视觉检查。

本轮已完成历史、代码、现有结果和基线测试的审计。真实数据输入、锁定环境、完整敏感度实验及更新海报的运行验收将在上述实现阶段完成。方案的主要自主设计决定为共同测试时间窗口、固定权重上限主实验、正权重分位数扩展、每个子样本重新 cross-fit，以及 50 次重复的初始计算规模。
