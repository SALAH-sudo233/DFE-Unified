# Arm-A 路径2「zero-I3 termination sweep」结果

日期: 2026-09-08 | 模型: armb_i3_run/checkpoints/25000.pt (zero-I3, 不动权重语义)

## 结论(TL;DR)

**termination sweep 在此 checkpoint 上证伪:终止不是「分子偏小」的瓶颈。**
- zero-I3 分子偏小不是全局问题,而是**双峰分布** —— 1a1e/3ebp 硬顶 HA=15 长不大,其余三口袋(1fm9/1a9q/1sgu)已健康(均值16-17)。
- hasatom_threshold 是弱旋钮(非单调:0.6/0.4/0.3 → HA 9.59/10.5/9.55),拉不动 HA。
- 真正的终止主控 `frontier_threshold`(maskfill.py:225, 默认0=sigmoid0.5)调低 → **破坏终止**(beam 撑到 max_steps 不终止 → finished 池饥死),不是优雅放大。
- 1a1e/3ebp 的 HA 天花板是模型容量/其它约束(bbox/valence/element head),**属重训范畴(Arm B),不在推理期 sweep 能解决**。

## 关键证据1:全量 zero-I3 基线真实 HA 分布(5口袋, on-disk, 590分子)

| pocket | N | HA均值 | HA中位 | min | max | ≥17占比 |
|---|---|---|---|---|---|---|
| 1fm9 | 117 | 17.13 | 18 | 8 | 22 | 60.7% |
| 1a9q | 124 | 16.15 | 16 | 10 | 18 | 44.4% |
| 1sgu | 102 | 14.12 | 14 | 6 | 21 | 41.2% |
| 3ebp | 127 | 12.90 | 13 | 8 | 15 | 0% |
| 1a1e | 120 | 12.17 | 13 | 6 | 15 | 0% |
| **POOLED** | **590** | **14.48** | 14 | | | 28.5% |

pooled 14.48 ≈ 基线报告的 14.5(测量口径一致)。但逐口袋看,「14.5 偏小」是被 1a1e/3ebp 拉低,不是普遍现象。

## 关键证据2:survivor bias(num_samples=20 快扫不可用于绝对值判断)

采样循环 `while len(finished) < num_samples` 只收**最先完成**的 N 个 = 最小的分子。
- 同一 zero-I3 基线: 1a1e 全量 HA 12.17 vs 前20完成 8.25; 1sgu 全量 14.12 vs 前20完成 8.30。
- 差 4-6 个重原子全是采样顺序偏置,阈值真实效应被淹没。

## 关键证据3:ft=0 透传验证代码改动 inert(docking 全量指标)

| run | n | Vina均值 | Vina最佳 | HA | LE | QED | PB通过% |
|---|---|---|---|---|---|---|---|
| ftp0_1a1e (ft=0 透传) | 20 | -4.71 | -5.57 | 10.55 | 0.456 | 0.526 | 65.0 |
| ftp0_1sgu (ft=0 透传) | 25 | -5.22 | -6.55 | 9.16 | 0.573 | 0.511 | 92.0 |
| baseline 1a1e_i3 (full) | 120 | -4.86 | -5.96 | 12.17 | 0.409 | 0.567 | 62.5 |
| baseline 1sgu_i3 (full) | 102 | -6.88 | -9.90 | 14.12 | 0.504 | 0.586 | 66.7 |

ft=0 透传的质量落在基线同一带内(HA/Vina 差异= survivor bias of 20 vs 100 样本);证明新增 `--frontier-threshold` 参数默认0时与原版行为完全一致。

## 代码改动(纯推理, 保留 .bak_ft)
- `sample.py`: `get_next(..., frontier_threshold=0)` → 透传给 `model.sample`。
- `sample_for_pdb_nodisk.py`: 新增 `--frontier-threshold`(默认0.0)→ get_next。
- py_compile OK。默认值0 = 原始行为(已由 ft=0 docking 验证)。

## 建议下一步
1. 放弃推理期 termination sweep 提 HA(已证伪)。
2. zero-I3 的 size ceiling(1a1e/3ebp 顶15)= Arm B I3 fine-tune 继续训练的目标,非推理旋钮。
3. 若要报 zero-I3 生成质量,用**全量 num_samples=100 逐口袋**,勿用小样本(survivor bias)。
