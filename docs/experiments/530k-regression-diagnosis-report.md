# DFE-Unified 530k 退化诊断报告

## 范围与限制

本报告只读检查已有训练日志、配置、checkpoint metadata，以及已存在的 CrossDocked diagnosis pilot 汇总；没有重新运行远程采样、Vina、训练或修改 main。

## 使用文件

- `/workspace/ayb/Pocket2Mol/logs/checkpoints/500000.pt`
- `/workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/checkpoints/530000.pt`
- `/workspace/ayb/Pocket2Mol/logs/train_df_2026_08_21__16_00_50/train_df.yml`
- `/workspace/ayb/Pocket2Mol/configs/train_trackA_ft.yml`
- `/workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/train.log`
- `/workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/run.sh`
- `/workspace/ayb/Pocket2Mol/train_resume.py`
- `/workspace/ayb/Pocket2Mol/utils/train.py`
- `/workspace/ayb/experiments/dfe-unified-cd100/diagnosis_pilot_v2/aggregate_out/summary.json`

## 已确认事实

### 1. CrossDocked pilot 的可复核结果

聚合器严格接受 72/72 个 run：12 个 manifest_idx × 2 model × 3 seed；没有缺失或被拒绝 entry。

| 指标 | 530k − df500k 配对均值 | bootstrap 95% CI |
|---|---:|---:|
| Vina mean | +0.1922 kcal/mol | [+0.0335, +0.3544] |
| Vina median | +0.2196 kcal/mol | [+0.0265, +0.4146] |
| Vina best | +0.2190 kcal/mol | [+0.0153, +0.4437] |
| direct PB pass rate | −0.0549 | [−0.1088, −0.0036] |
| QED | −0.0031 | [−0.0156, +0.0090] |
| heavy atoms | −0.4445 | [−0.9163, −0.0164] |
| ligand efficiency | +0.0035 | [−0.0062, +0.0140] |
| unique ratio | 0 | [0, 0] |

这里的 delta 定义为 `530k - df500k`；Vina 越负越好，因此正 delta 表示 530k 变差。该 pilot 数字不是用户描述的 93-pocket preliminary 全量结果，二者样本范围不同。

### 2. 配置对比

| 项目 | 500k baseline | 530k fine-tune | 事实解释 |
|---|---|---|---|
| checkpoint iteration | 500000 | 530000 | 530k 从 500k resume，增加 30000 steps |
| batch size | 8 | 8 | 未变 |
| optimizer | Adam | Adam | 未变；beta=(0.99,0.999)，weight decay=0 |
| YAML lr | 2e-4 | 1e-4 | 新配置声明减半 |
| checkpoint 实际 lr | 1e-5 | 1e-5 | 530k 继承 checkpoint optimizer state，未使用 YAML 的 1e-4 作为最终实际 lr |
| scheduler | ReduceLROnPlateau，factor=0.6，patience=8，min_lr=1e-5 | 相同 | scheduler state 也被恢复 |
| YAML val frequency | 5000 | 3000 | 530k 更密集记录验证 |
| max gradient norm | 100 | 100 | 未变 |
| noise std | 0.1 | 0.1 | 未变 |
| mask/contrastive transform | 相同 | 相同 | 未见数据变换改变 |

`train_resume.py` 的实际顺序是：创建 optimizer/scheduler → `model.load_state_dict` → `optimizer.load_state_dict(ckpt['optimizer'])` → `scheduler.load_state_dict(ckpt['scheduler'])`。因此 optimizer/scheduler checkpoint state 覆盖了新 YAML 对应的初始 state。

### 3. checkpoint metadata

| checkpoint | iteration | optimizer lr | scheduler last_epoch | scheduler best | num_bad_epochs |
|---|---:|---:|---:|---:|---:|
| 500000.pt | 500000 | 1e-5 | 100 | 1.0497096204 | 4 |
| 530000.pt | 530000 | 1e-5 | 111 | 1.0497096204 | 6 |

530k 的 scheduler `last_epoch=111` 与 500k 的 `100` 相差 11 次验证；这是由 530k `val_freq=3000`、每 3000 step 调 scheduler 所致。没有证据表明 LR 在 500k→530k 期间回升；最低 LR 仍为 `1e-5`。

### 4. 530k 验证 loss 序列

| iter | val total | Fron | Pos | Cls | Edge | Real | Fake | Surf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 501k | 1.8280 | 0.0856 | 0.7805 | 0.7260 | 0.0270 | 0.0675 | 0.1036 | 0.0379 |
| 504k | 2.5954 | 0.0524 | 1.5367 | 0.7695 | 0.0292 | 0.0554 | 0.1056 | 0.0466 |
| 507k | 2.4867 | 0.0583 | 1.4914 | 0.7185 | 0.0226 | 0.0470 | 0.0774 | 0.0715 |
| 510k | 1.6159 | 0.0524 | 0.5646 | 0.7923 | 0.0415 | 0.0447 | 0.0922 | 0.0283 |
| 513k | 1.2303 | 0.0598 | 0.2356 | 0.7331 | 0.0196 | 0.0382 | 0.1056 | 0.0384 |
| 516k | 1.2693 | 0.0955 | 0.2583 | 0.7053 | 0.0438 | 0.0390 | 0.0855 | 0.0420 |
| 519k | 1.7165 | 0.0819 | 0.6946 | 0.7107 | 0.0448 | 0.0441 | 0.1095 | 0.0310 |
| 522k | 1.3226 | 0.0553 | 0.3353 | 0.7583 | 0.0238 | 0.0361 | 0.0858 | 0.0279 |
| 525k | 1.2101 | 0.0475 | 0.2615 | 0.7418 | 0.0215 | 0.0314 | 0.0783 | 0.0281 |
| 528k | 2.2763 | 0.0627 | 1.2904 | 0.7133 | 0.0294 | 0.0366 | 0.0901 | 0.0538 |
| 530k | 1.3715 | 0.0699 | 0.3987 | 0.7187 | 0.0245 | 0.0435 | 0.0903 | 0.0259 |

### 5. loss 是否有突变/过拟合

- 530k 训练日志从 500001 开始，最初若干 batch 的 total loss 在约 −1.6 到 3.9 间波动，Pos 约 −2.3 到 3.0；末段也存在类似的 batch-level 波动。这是当前 stochastic loss 定义下的现象，不能单凭逐 batch 值称作数值爆炸。
- 验证 total loss 在 501k、504k、507k、519k、528k 有明显回升，随后又下降；没有单调恶化。
- Pos validation 在 501k–507k 偏高，513k–516k 降到约 0.24–0.26，528k 又升到 1.29，530k 回到 0.40。该非单调性说明训练过程不稳定/高方差，但不是典型“持续训练 loss 降、validation loss 单调升”的清晰过拟合曲线。
- Cls、Edge、Real、Fake 没有显示 530k 末端同步爆炸；Fron 也没有持续恶化。
- 训练日志没有 baseline 500k 的等频 validation 序列可直接与 530k 对齐；因此“530k validation 比 500k 变差多少”目前不能从日志精确计算。

## 可能原因（证据支持程度）

### 高可信：继续更新了生成模型，但没有以 baseline 保持为约束

530k 是全模型继续训练，不是只更新某个 head。30k steps 即使实际 LR 已降至 1e-5，仍足以改变 Position/field/frontier 等生成行为；CrossDocked pilot 的 heavy atoms 下降、Vina 变差、PB 轻微下降与生成分布漂移一致。当前证据支持“全模型继续更新造成了 baseline 行为漂移”，但不能仅凭日志定位到某一个 head。

### 中高可信：Pos 分支是最可疑的退化通道

验证 Pos 在早期 501k–507k 较高，513k–516k 曾改善，528k 又出现 1.2904 的回升；Position 是直接影响坐标与后续可生成几何的分支。该模式与 CrossDocked 生成质量下降相容，但还没有参数差分或固定 batch 的输出对照，不能称为已证实根因。

### 中可信：scheduler state 与新 fine-tune YAML 的语义不一致

530k YAML 写 `lr=1e-4`，但 resume 后实际 optimizer lr 为 `1e-5`；scheduler 也继承旧的 plateau history。若实验意图是“从 500k 以 1e-4 做 30k fine-tune”，实际实验并没有执行该协议。它可能造成更新过小、进入高方差的局部漂移，或让用户对实验设定产生误判；但“最低 LR 本身导致 Vina 下降”尚无直接证据。

### 中可信：训练目标与生成质量不完全一致

训练 log 只优化既有生成/对比损失，没有 raw Vina、ligand-efficiency 或 direct PB 目标。Loss 降低不保证生成候选的 docking 变好；尤其 Pos loss、Fron loss 与 Vina 的映射并未验证。需要固定样本/固定 pocket 的 checkpoint trajectory 才能判断 loss 改善是否伴随生成退化。

## 尚无证据的假设

- 不是已确认的“学习率过大导致 catastrophic forgetting”：实际 lr 是 1e-5，日志没有 LR spike。
- 不是已确认的“经典过拟合”：验证 loss 非单调持续上升，且缺少 500k 同协议验证基线。
- 不是已确认的“只因分子更小”：pilot 中 heavy atoms 确实平均下降，但 ligand efficiency delta 近零且 CI 跨 0；尺寸分层仍需以完整结果判断。
- 不是已确认的 frontier head、Atom head、Bond head 或 DF field 单独退化；现有日志没有参数/输出漂移分解。
- 不能排除固定 seed、数据顺序、训练恢复状态、checkpoint 评估代码差异造成的部分影响。

## 最值得做的两个验证实验（不建议先扩大训练）

### 实验 1：同 batch、同输入的 checkpoint trajectory

对 500000.pt、528000.pt、530000.pt 使用完全相同的固定 train/validation batch，逐项记录 `Loss(Fron/Pos/Cls/Edge/Real/Fake/Surf)` 以及关键 head 输出的差值。优先比较：Position 的 `abs_pos_mu/pos_sigma`、frontier logits、element/hasatom logits、bond logits、DF field 向量范数/方向。

Go/no-go：若在固定 batch 上 Pos 或 frontier/position 输出从 500k 到 530k 显著漂移，而其他项稳定，则可把生成退化归因进一步缩小到该分支；若所有 head 输出均小幅变化，则优先检查 sampling/评测协议与随机性。

### 实验 2：只读 checkpoint 的小型配对生成轨迹（不重新跑 Vina）

在已有生成结果或允许的最小本地 fixture 上，对 500k、528k、530k 做同 pocket、同 seed、同采样配置的生成对照；只先统计 completion/termination、heavy atoms、validity、前沿存活和结构级几何指标。不要先用扩大样本量掩盖问题。

Go/no-go：若 530k 在同一 seed 下出现 heavy-atom 左移、termination/validity 改变，同时固定 batch 的 Position/frontier 输出也漂移，支持生成头漂移；若生成统计不变而 Vina 变化，优先审查 evaluator 或 docking-box/输入协议。

## 最终诊断

当前最稳妥的结论是：**530k 退化发生在一次全模型 resume fine-tune 后，伴随生成尺寸下降和 Pos validation 的高方差/回升；配置实际使用的是从 500k checkpoint 继承的最低 LR 1e-5，而不是 YAML 写的 1e-4。日志支持“全模型继续更新造成生成分布漂移”，但尚不能把责任唯一归于 Pos、Frontier、LR 或经典过拟合。**

不建议继续用 530k 作为优化目标，也不建议直接把 30k continuation 延长。先做固定 batch trajectory 与 checkpoint 生成对照；后续微调应重置 optimizer/scheduler、使用更短步数/更低有效 LR，并保留 baseline replay 或部分冻结。
