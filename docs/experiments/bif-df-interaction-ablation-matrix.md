# BIF+DF interact / conservative optimization ablation matrix

日期：2026-09-14

## 固定约束

- 不延长 530k continuation。
- 不使用 530k 作为训练起点。
- 所有训练从原始 DF-500k `500000.pt` 开始，独立输出目录。
- 不把 raw Vina 作为唯一目标；每个候选必须同时记录 raw Vina、ligand efficiency、heavy atoms、QED、direct PoseBusters、有效产物数/终止率、per-pocket macro-average 和 seed 方差。
- 12-pocket pilot：复用 `diagnosis_pilot_v2` 的 pocket、seed 和采样协议。
- acceptance：paired Vina 不劣于 baseline 超过 0.10 kcal/mol；LE 不下降；direct PB 不下降超过 2 pp；heavy-atom 分布无异常漂移。

## 阶段 0：已完成的诊断

- 已完成 500k/528k/530k checkpoint 参数级初筛。
- 最大漂移集中于 encoder interaction/message path；不是 frontier/position head 主导。
- 该分析尚不是固定 batch trajectory，也不能单独证明因果。

## 阶段 1：固定 batch trajectory

同一 batch、同一随机种子、eval mode，对 500k/528k/530k 输出逐层记录：

- encoder scalar/vector；
- field/DF；
- frontier；
- position mu/sigma；
- atom/element；
- bond/edge；
- interaction/pair residual（如模块存在）。

输出：每层 absolute delta、relative delta、cosine similarity、分布统计和 JSON evidence。

## 阶段 2：model-only warm-start + 方案 A dry-run

起点：500k。

- 冻结 encoder + field/DF；
- 只训练生成 output heads；
- 重置 optimizer；
- lr=1e-5；
- 1-step load/forward/backward/save smoke；
- 确认 trainable parameter names，不接受只打印数量。

## 阶段 3：A 的 12-pocket pilot

- 仅在 dry-run 全部通过后执行；
- 12 pockets × 3 seeds；
- 与 baseline 严格配对；
- 先评估 1k/2k/5k checkpoint，禁止直接只看最终 checkpoint。

## 阶段 4：方案 B

- 冻结 encoder；
- 解冻 field/DF + output heads；
- 其他设置与 A 相同；
- 先 smoke，再 12-pocket pilot。

## 阶段 5：方案 E

仅当 A 达到 acceptance：

- baseline replay + 部分冻结；
- 保留原始训练目标，同时加入经过验证的 replay 约束；
- 先做小规模 dry-run，不直接全量训练。

## 阶段 6：BIF frozen probe

生成模型完全冻结，不做 joint fine-tune。比较：

- full pair representation；
- ligand-only；
- pocket-only；
- pocket shuffle；
- geometry shuffle；
- label permutation sanity check。

只有 full pair 显著超过所有 control，才进入 interaction head joint fine-tune 设计。

## 当前状态

- 530k continuation：停止，不复用。
- conservative FT 5k：已完成训练，但尚未作为 A/B 候选接受，必须重新确认其冻结集合与协议对应关系后再评测。
- fixed-batch trajectory：参数级初筛完成，固定 batch forward trajectory 尚未完成。
- model-only runner：需要单独落盘并完成 dry-run 证据。
- BIF frozen probe：既有 P3 结果显示 full=0.648、ligand-only=0.628、pocket-only=0.631、pocket-shuffle=0.611；pair-specific signal 很弱，需 geometry-shuffle 复核。
