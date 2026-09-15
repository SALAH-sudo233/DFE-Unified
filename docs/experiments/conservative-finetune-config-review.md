# DFE-500k 保守微调配置审查

## 总结判断

优先级：**E > A > B > C > D**。

- 530k 的 resume 实际继承了 500k checkpoint 的 optimizer/scheduler state，实际 LR 为 `1e-5`，且全模型更新后 CrossDocked 质量下降；因此下一次不应直接沿用这种 resume 语义。
- 必须明确“重置 optimizer/scheduler”还是“恢复 optimizer/scheduler”。保守微调目标是控制参数漂移，不是复现 530k。
- 所有候选都必须单独 logdir、单独 checkpoint，不得覆盖 `500000.pt`、`530000.pt` 或 `armb_i3_run/checkpoints/25000.pt`。

## 方案审查

| 方案 | 设计 | checkpoint 加载 | 建议 |
|---|---|---|---|
| A | 5k steps，lr=1e-5，只 output heads | 模型 state 可严格加载；需要定义 heads 为 `frontier_pred`、`field.classifier/edge_pred`、`pos_predictor`，不要误把 encoder 归入 head | 首轮推荐 |
| B | 10k，lr=1e-5，heads + position | 结构不变，可严格加载；position 训练风险高于 A | 第二顺位 |
| C | 20k，lr=1e-5，冻结 encoder 前4层 | checkpoint 可加载；需实现按名称冻结 `encoder.interactions.0–3`，但 field/position 仍可漂移 | 第三顺位 |
| D | 30k，lr=5e-6，全模型，strict clip | 可加载，但仍是全模型漂移；`max_grad_norm=100` 不是强约束，建议真实使用 1.0 或 5.0 并记录 grad norm | 不作为首轮 |
| E | baseline replay + 部分冻结 | 模型可加载；需要训练数据/采样 replay 设计，避免只在 CrossDocked continuation 上拟合 | 最值得作为稳健主线 |

## optimizer/scheduler 建议

### 首轮 A/B

- 从 `500000.pt` 只加载 `model`。
- **重置 Adam optimizer**，不要加载 checkpoint 的 optimizer state；否则会把 `lr=1e-5`、momentum/二阶矩和 plateau 历史带入新实验。
- 使用固定 cosine/constant-with-warmup 或无 scheduler 的短程方案；不建议短程 5k/10k 使用 plateau，因为 validation 次数太少且 scheduler history 没有稳定意义。
- 新实验 LR 以 optimizer 初始化后的实际值为准，启动时打印并断言 `lr==1e-5`。

### C/D/E

- C：冻结参数后再构造 optimizer，只把 `requires_grad=True` 的参数传给 Adam；否则“冻结”但仍进入 optimizer 会增加状态/审计复杂度。
- D：全模型 + `lr=5e-6`，重置 optimizer；每 step 记录 unclipped/clipped grad norm，严格 `error_if_nonfinite=True`。
- E：重置 optimizer；每个 batch 混合当前训练样本与 replay 样本，generation loss 保持主权重，不能只优化辅助目标。

## 建议独立配置草案

下面配置假设 runner 支持 `resume_model_only`、`freeze_patterns`、`replay_ratio` 等字段；这些字段不能直接传给当前未修改的 `train_resume.py`，运行前必须在独立 runner 中实现并做 dry-run。不得直接拿它覆盖现有配置。

### A: head-only 5k

```yaml
model:
  checkpoint: /workspace/ayb/Pocket2Mol/logs/checkpoints/500000.pt
train:
  seed: 20260914
  max_iters: 505000
  batch_size: 8
  resume_model_only: true
  reset_optimizer: true
  optimizer:
    type: adam
    lr: 1.0e-5
    weight_decay: 0
    beta1: 0.99
    beta2: 0.999
  scheduler:
    type: none
  max_grad_norm: 1.0
  val_freq: 1000
  save_freq: 1000
  trainable_patterns:
    - frontier_pred
    - field.classifier
    - field.edge_pred
    - pos_predictor
```

### B: heads + position 10k

```yaml
model:
  checkpoint: /workspace/ayb/Pocket2Mol/logs/checkpoints/500000.pt
train:
  seed: 20260914
  max_iters: 510000
  batch_size: 8
  resume_model_only: true
  reset_optimizer: true
  optimizer: {type: adam, lr: 1.0e-5, weight_decay: 0, beta1: 0.99, beta2: 0.999}
  scheduler: {type: none}
  max_grad_norm: 1.0
  val_freq: 1000
  save_freq: 1000
  trainable_patterns: [frontier_pred, field.classifier, field.edge_pred, pos_predictor]
```

### C: 前4层 encoder 冻结 20k

```yaml
model:
  checkpoint: /workspace/ayb/Pocket2Mol/logs/checkpoints/500000.pt
train:
  seed: 20260914
  max_iters: 520000
  batch_size: 8
  resume_model_only: true
  reset_optimizer: true
  optimizer: {type: adam, lr: 1.0e-5, weight_decay: 0, beta1: 0.99, beta2: 0.999}
  scheduler: {type: none}
  max_grad_norm: 1.0
  val_freq: 2000
  save_freq: 2000
  freeze_patterns: [encoder.interactions.0, encoder.interactions.1, encoder.interactions.2, encoder.interactions.3]
```

### D: 全模型 30k，低 LR + 强 clip

```yaml
model:
  checkpoint: /workspace/ayb/Pocket2Mol/logs/checkpoints/500000.pt
train:
  seed: 20260914
  max_iters: 530000
  batch_size: 8
  resume_model_only: true
  reset_optimizer: true
  optimizer: {type: adam, lr: 5.0e-6, weight_decay: 0, beta1: 0.99, beta2: 0.999}
  scheduler: {type: none}
  max_grad_norm: 1.0
  val_freq: 2000
  save_freq: 2000
```

### E: baseline replay + 部分冻结

```yaml
model:
  checkpoint: /workspace/ayb/Pocket2Mol/logs/checkpoints/500000.pt
train:
  seed: 20260914
  max_iters: 515000
  batch_size: 8
  resume_model_only: true
  reset_optimizer: true
  optimizer: {type: adam, lr: 1.0e-5, weight_decay: 0, beta1: 0.99, beta2: 0.999}
  scheduler: {type: none}
  max_grad_norm: 1.0
  val_freq: 1000
  save_freq: 1000
  replay_ratio: 0.5
  freeze_patterns: [encoder.interactions.0, encoder.interactions.1, encoder.interactions.2, encoder.interactions.3]
```

这些是配置草案，不代表当前 runner 已支持所有字段；在任何训练前必须做 `--dry-run`：实例化模型、加载 500k model state、打印 missing/unexpected keys、打印 trainable parameter count、打印实际 optimizer LR，并退出。

## 12-pocket pilot

- 固定同一 12 个 `manifest_idx`，每个候选 checkpoint 使用相同 3 seeds：2020/2021/2022。
- 每 pocket 每 seed 生成数量、beam、max_steps、center、pocket10/receptor、Vina evaluator 全部固定。
- 先做生成有效性与 termination 统计，再做 clean Vina/PB；不要用不同分母的“成功分子均值”。
- 与 500k baseline 成对比较；530k 只作为退化参考，不作为优化目标。
- 统计 macro pocket mean、seed mean/SD、paired pocket delta 和 bootstrap CI。

## go/no-go 门槛

以 500k 为参照，候选必须同时满足：

1. paired raw Vina delta ≤ 0；若 CI 上界明显 >0，停止。
2. ligand-efficiency delta ≥0；不能靠增加 heavy atoms 换 raw Vina。
3. direct PoseBusters pass rate 不低于 baseline，允许误差需预注册，例如不超过 2 percentage points。
4. heavy atoms 与 baseline 差异控制在预设窗口内，例如绝对差不超过 1.0；不得把尺寸变化当成功。
5. valid/termination rate 不下降；若出现 completion/termination 崩溃，即使 Vina 变好也停止。
6. 三个 seeds 方向至少一致，不能由单 seed 贡献全部增益。

建议把“超过 baseline”定义为 paired CI 支持，而非只比较点估计；12 pockets 仅作 pilot，不足以支撑最终 SOTA 结论。

## 风险

- 当前 `train_resume.py` 无条件加载 `optimizer` 和 `scheduler` state；仅改 YAML 的 LR 不足以改变恢复后的实际 LR。
- `max_grad_norm=100` 在 530k 配置中不是严格的防漂移 clip；保守实验应改为 1.0 或 5.0，并记录原始 norm。
- C 的“冻结前4层”必须确认命名匹配 `encoder.interactions.0`–`.3`；不能只按参数数量猜测。
- 方案 A 的“output heads”定义必须以 state-dict 前缀和 forward 路径核实；`field` 内部包含生成分类/edge 组件，不能只冻结名为 `classifier` 的一小部分。
- 每个方案都需要独立 output directory；任何 checkpoint 保存策略都不能指向已有目录。

## 建议优先级

1. 先做 A 的 dry-run 和 12-pocket pilot。
2. 若 A 保住 baseline 但无改善，做 E；E 是防灾难性遗忘的首选。
3. B 用于判断 position head 是否是必要自由度。
4. C 用于判断 encoder 漂移是否是主要来源。
5. D 最后做；若 A/E 已失败，不应直接用 D 延长全模型训练。
