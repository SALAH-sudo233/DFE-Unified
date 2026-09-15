# CrossDocked-100 退化诊断与优化实验设计

日期：2026-09-11
状态：诊断已完成；新实验仅完成设计，尚未启动训练。

## 1. 当前诊断结论

基于 93 个唯一输出目录、两模型均已完成 clean eval：

- 530k：9,836 条分子，Vina macro-pocket mean = -6.704，direct PoseBusters = 62.1%，平均 heavy atoms = 15.29。
- DF-500k baseline：10,032 条分子，Vina macro-pocket mean = -6.950，direct PoseBusters = 60.8%，平均 heavy atoms = 16.03。
- 配对 pocket：530k 优于 baseline 的 pocket 为 26/93，劣于 baseline 为 67/93；平均差（530k - baseline）= +0.246 kcal/mol。
- checkpoint 配置显示 530k 继续训练 30k steps，batch=8，Adam lr=1e-4；baseline 为 500k checkpoint，lr=2e-4。两者不是同一训练预算下的简单随机波动。
- 尺寸分层后，530k 在 21+ heavy atoms 区间 Vina = -8.72，baseline = -8.94；小尺寸区间几乎相同。因此尺寸偏移是部分原因，但不能解释全部退化。
- 530k 的分子数和终止率更低，说明微调改变了生成/终止分布；不能只归因于 docking box 或 PoseBusters。

## 2. 必须先完成的诊断

### D0：结果完整性与配对统计
- 修复重复目录 key：输出 key 使用 `manifest_idx + pocket10 basename`，不要使用 target 目录名。
- 补跑 7 组重复 manifest entry；最终按 100 entries 聚合。
- 预注册主指标：每-pocket macro Vina、micro Vina、LE、direct PB、QED、heavy atoms、有效产物数、强 docking 比例。
- 对配对差使用 pocket bootstrap 95% CI；不以 pooled molecule 数量替代 pocket-level 统计。

### D1：采样随机性诊断
- 从 12 个 pocket 分层抽样：低/中/高 baseline Vina 各 4 个。
- baseline 与 530k 使用完全相同 seed 集合 `{2020, 2021, 2022}`，每次 30 个样本，保持 bbox/beam/max_steps/Vina 配置不变。
- 目标：估计 seed 方差，判断 +0.246 是否超过采样噪声；比较每个 seed 的配对差，而不是只比较一次 100 分子池。

### D2：生成分布漂移
逐 pocket、逐模型记录：
- 有效率、失败率、终止步数；
- 分子 heavy atoms、键数、环数、HBD/HBA、formal charge、QED、logP；
- unique ratio、重复 SMILES 频率；
- atom/element/bond/frontier 终止类别分布。

重点检验：530k 是否出现更强的小分子偏置、重复率上升或 frontier 终止改变。

### D3：尺寸匹配与 interaction proxy
- 在每个 pocket 内按 heavy-atom bins `[8-11], [12-14], [15-17], [18-20], [21+]` 重采样。
- 同时报告 raw Vina、`-Vina/HA`、尺寸匹配 Vina 和 PB。
- 使用 reference ligand 仅作为尺寸/化学性质描述，不将其 docking 分数混入生成分子均值。
- 若尺寸匹配后仍显著退化，则确认是 pocket-specific interaction/几何分布退化，而非单纯尺寸问题。

### D4：checkpoint trajectory
如果保留有 500k 之后的 checkpoint，选择 505k/510k/520k/530k；否则不补造。
- 每个 checkpoint 在 12 pocket × 1 seed 上做快速 30 分子筛查。
- 画 Vina、LE、PB、HA、有效率随训练步数曲线。
- 目标：定位退化发生在微调初期还是持续累积。

### D5：训练/验证指标关联
从 `train.log` 提取每次 validation 的总 loss、Pos/At/Edge/Frontier/DF 相关 loss 和学习率；与 D4 的生成指标对齐。
- 若 validation loss 改善而 Vina 恶化：确认生成似然与 affinity 不一致。
- 若某一头 loss/梯度异常：优先做头部冻结或分层学习率实验。

## 3. 新优化实验：分阶段，不盲目重训

### O0：回退与短步数校准（最高优先级）
从原始 DF-500k baseline 出发，不覆盖已有 checkpoint：

- `FT-5k`: 5k steps，lr=1e-5，encoder/field 冻结，仅训练输出头；
- `FT-10k`: 10k steps，lr=1e-5，输出头 + position head；
- `FT-30k-lowLR`: 30k steps，lr=1e-5，全模型但 gradient clip=1；
- 每个版本先在 D1 的 12 pocket × 3 seeds 上评测，不通过就停止扩展。

判据：配对 Vina 不得比 baseline 差超过 0.10 kcal/mol；PB/LE/有效率不能同时恶化。

### O1：保守微调（推荐主线）
目标是避免灾难性遗忘，而不是单纯继续拟合。

- baseline checkpoint warm-start；
- encoder 前 4 层冻结；只训练 field、position 和 frontier/atom/bond 输出头；
- lr=1e-5；EMA 或每 2k steps 保存一次；
- 加入 baseline replay：每个 batch 保留一部分原始训练分布样本；
- 训练 5k/10k/20k 三个 checkpoint。

需要同时记录生成分布漂移，不能只看训练 loss。

### O2：尺寸/终止校准（低成本诊断，不声称解决 affinity）
只在 O0/O1 确认主要问题是尺寸和终止分布时进行：

- 对 frontier/no-frontier 与 molecule-size 相关 loss 加小权重校准；
- 目标尺寸来自 baseline 分布的 pocket-conditional histogram，而不是人为追求更大分子；
- 不直接优化 raw Vina，不把尺寸奖励当作 interaction 学习；
- 先做 12 pocket 小评测，防止 size gaming。

### O3：interaction-aware 辅助目标（真正冲 SOTA 的主实验）
如果 D3 证明尺寸匹配后仍退化，才进入此实验。

- 保持生成主干和采样协议不变；
- 从蛋白-配体局部几何构造 pairwise interaction head：距离、方向、元素/芳香性、H-bond donor/acceptor 等；
- 使用训练配体的正样本和 pocket 内几何扰动/错配负样本；
- auxiliary loss 权重从 `0.05/0.1/0.2` 网格开始；
- 必须有 ligand-only、pocket-only、pocket-shuffle 和 size-matched controls；
- 评测以 docking + LE + PB + interaction proxy 联合判断，避免仅提升 raw Vina。

### O4：生成后筛选作为独立上限实验
不改生成器，使用固定生成池进行：

- baseline 与 530k 各自的同 pocket pool；
- 仅使用训练集可获得的模型分数进行 rerank；
- 报告 rerank 前后 raw Vina、LE、PB、HA 和 diversity；
- 禁止使用 Vina 作为选择器后再报告同一 Vina 的“提升”；
- 需要独立 docking/held-out scorer 或 cross-validation 才能证明筛选有效。

## 4. 实验停止与成功标准

### 停止条件
- 12-pocket pilot 的 paired Vina 95% CI 明显劣于 baseline；
- 530k 继续出现有效率/终止率下降；
- PB 和 LE 同时下降；
- 只靠增大 heavy atoms 才能获得 raw Vina 改善。

### 进入全量 100-entry 的条件
- pilot 中 paired Vina 至少不劣于 baseline 0.10 kcal/mol；
- 尺寸匹配后仍保持正向或中性；
- direct PB、LE、diversity 没有明显退化；
- seed 方差低于观察到的收益；
- 输出目录和 manifest entry 一一对应。

### 不接受的结论
- 不把单次 PDBBind 结果当 CrossDocked SOTA 证据；
- 不把 raw Vina 提升当作 interaction 学习证据；
- 不把 warm-start ablation 当作从头训练的纯因果消融；
- 不把 direct `PoseBusters(config='mol')` 与 redocked-pose PB 混为同一指标；
- 不在 100-entry 修复前发布最终排名。

## 5. 建议执行顺序

1. 修复重复 entry 输出目录并补齐 100-entry 数据；
2. 先跑 D1 + D2 + D3 的 12-pocket pilot；
3. 同时从 train.log 提取 D5 曲线；
4. 根据诊断结果选择 O0/O1 或 O3，不同时启动多个大训练；
5. 通过 pilot 门槛后再启动 100-pocket 评测；
6. 保留 baseline、530k 和每个候选 checkpoint，独立目录，不覆盖 `armb_i3_run/checkpoints/25000.pt`。
