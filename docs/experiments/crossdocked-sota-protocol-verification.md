# CrossDocked-2020 SOTA 评测协议逐项核验

## 结论

四个方法都以 CrossDocked-2020 测试 pocket 为核心，但不能直接把论文中的 Vina 数字排成同一张严格可比表。关键差异包括：CrossDocked 版本/test split、receptor 文件、每 pocket 样本数、QVina/Vina 实现、score-only/minimize/dock 模式、exhaustiveness、聚合方式、reference ligand 和 size control。

原始官方评测代码中没有统一使用 PoseBusters；PoseBusters pass rate 不能替代原始 Success Rate。

## 官方来源

- Pocket2Mol README: https://github.com/pengxingang/Pocket2Mol/blob/main/README.md
- Pocket2Mol data README: https://github.com/pengxingang/Pocket2Mol/blob/main/data/README.md
- Pocket2Mol docking: https://github.com/pengxingang/Pocket2Mol/blob/main/evaluation/docking.py
- Pocket2Mol evaluator: https://github.com/pengxingang/Pocket2Mol/blob/main/evaluation/evaluate.py
- TargetDiff README: https://github.com/guanjq/targetdiff/blob/main/README.md
- TargetDiff diffusion evaluator: https://github.com/guanjq/targetdiff/blob/main/scripts/evaluate_diffusion.py
- TargetDiff metadata evaluator: https://github.com/guanjq/targetdiff/blob/main/scripts/evaluate_from_meta.py
- TargetDiff evaluation utilities: https://github.com/guanjq/targetdiff/tree/main/utils/evaluation
- TargetDiff summary notebook: https://github.com/guanjq/targetdiff/blob/main/notebooks/summary.ipynb
- DecompDiff README: https://github.com/bytedance/DecompDiff/blob/main/README.md
- DecompDiff evaluator: https://github.com/bytedance/DecompDiff/blob/main/scripts/evaluate_mol_from_meta_full.py
- DecompDiff evaluation utilities: https://github.com/bytedance/DecompDiff/tree/main/utils/evaluation
- MolCRAFT paper: https://arxiv.org/html/2404.12141v4
- MolCRAFT proceedings: https://proceedings.mlr.press/v235/qu24a.html
- MolCRAFT repository: https://github.com/GenSI-THUAIR/MolCRAFT
- CrossDocked source: https://bits.csb.pitt.edu/files/crossdock2020/

## 1. Test split 与数据版本

| 方法 | 官方可核对信息 |
|---|---|
| Pocket2Mol | README 的 `sample.py --data_id {i}` 使用 `0–99`，即 100 个测试 pocket；使用 `split_by_name.pt`/pocket10 数据。 |
| TargetDiff | README 要求 CrossDocked2020 v1.1、RMSD < 1 Å 清洗版本；`evaluate_from_meta.py` 默认 `eval_num_examples=100`，使用 `test_index[:100]`。 |
| DecompDiff | README 要求 `split_by_name.pt`、`test_index.pkl` 和 `test_set.zip`；`evaluate_mol_from_meta_full.py` 默认 `eval_num_examples=100`，使用 `test_index[:100]`。 |
| MolCRAFT | 论文明确使用 CrossDocked2020；公开仓库没有完整公开 MolCRAFT-specific test-index/evaluation 参数，不能仅凭 README 证明其严格复用了同一 100-pocket 顺序。 |

## 2. receptor、pocket10 与 docking box

### Pocket2Mol

官方 `evaluation/docking.py` 的 `QVinaDockingTask`：

- 从 `protein_root` 读取 receptor PDB；
- 若没有显式中心，docking box 中心为配体坐标轴向包围盒中心：`(pos.max(0)+pos.min(0))/2`；
- docking box 为 `20 × 20 × 20 Å`；
- 默认 QVina exhaustiveness 为 16。

README 中的 23 Å bounding box 是采样输入 box，不等于 docking evaluator 的 20 Å box。

### TargetDiff

正式 test-set 命令使用：

```bash
python scripts/evaluate_diffusion.py OUTPUT_DIR \
  --docking_mode vina_score \
  --protein_root data/test_set
```

默认训练/处理路径与正式 test-set receptor root 不应混淆。box 的具体实现由 `VinaDockingTask` 和 evaluation utilities 控制，不能从 README 推断为 Pocket2Mol 的 20 Å box。

### DecompDiff

默认 `protein_root='./data/test_set'`，即使用 test_set 中的 receptor PDB。DecompDiff 的 box/docking 实现不能自动套用 TargetDiff 参数。

### MolCRAFT

公开论文可确认 CrossDocked pocket 条件和 controlled-size 评测，但无法从公开 README 完整核对 receptor 文件、box 中心、box 边长和 docking mode。

## 3. 每 pocket 生成/统计多少分子

| 方法 | 官方代码事实 |
|---|---|
| Pocket2Mol | README 目标是 100 test pockets；`evaluation/evaluate.py` 对 `samples.finished` 中实际成功样本逐个评估，脚本本身不统一截断为每 pocket 100 个。 |
| TargetDiff | `evaluate_from_meta.py` 对每个 pocket 使用 `index[:100]`。 |
| DecompDiff | `evaluate_mol_from_meta_full.py` 对每个 pocket 使用 `index[:100]`。 |
| MolCRAFT | 论文强调 controlled molecular sizes，但公开仓库未核对出每 pocket 截断逻辑。 |

因此“100 pocket”不等于“每 pocket 100 个有效分子”，尤其当生成失败、重构失败或多片段过滤存在时，最终分母可能不同。

## 4. Vina 模式与参数

| 方法 | Vina/评分实现 | 默认参数与含义 |
|---|---|---|
| Pocket2Mol | QVina docking | `QVinaDockingTask`；box 20 Å；exhaustiveness 16；取 docking 输出最佳 affinity。 |
| TargetDiff | `qvina`、`vina_score`、`vina_dock`、`none` | `evaluate_diffusion.py` 默认 exhaustiveness 16；`evaluate_from_meta.py` 的默认 `docking_mode=vina_full`、exhaustiveness 32。`vina_score` 是 score-only；`vina_dock` 额外执行 dock。 |
| DecompDiff | `none`、`vina_score`、`vina_full` | 默认 `vina_full`、exhaustiveness 32；流程为 score-only → minimize → dock，并保存三种结果。 |
| MolCRAFT | 论文报告 Vina Score | 公开仓库没有足够信息核对 score-only/minimize/dock、exhaustiveness、box 和 receptor preparation。 |

TargetDiff 的两个官方 evaluator 存在不同默认设置，不能只写“TargetDiff Vina”。

## 5. validity、success、PoseBusters

TargetDiff evaluator 区分：

- molecule stability；
- reconstruction success；
- complete molecule；
- evaluation success。

多片段分子通过 SMILES 中的 `.` 排除。DecompDiff README 的 Success Rate 是其自身综合成功定义，不等同于 validity 或 PoseBusters pass。

原始四个方法的官方主评测中没有核对到统一 PoseBusters 主指标。因此：

```text
complete molecule != PoseBusters pass
Success Rate != validity rate
Success Rate != direct PoseBusters pass rate
```

## 6. 聚合方式

TargetDiff summary notebook 对 Vina/QED/SA/size 等分子结果展平后做 pooled mean/median；High Affinity 则先按 pocket 计算比例，再对 100 个 pocket 的比例取均值/中位数。

DecompDiff evaluator 也将 per-pocket 结果展平后对成功样本计算 Vina/QED/SA 均值。因此其主表 Vina/QED/SA 是 pooled molecule-level aggregation，而不是 pocket-equal-weight macro-average。

Pocket2Mol 的单分子 evaluator 本身不足以推断论文表格的完整 pocket-level 聚合；需要结合其论文 supplementary 或汇总脚本。

## 7. size matching 与 ligand efficiency

MolCRAFT 论文明确强调：

> reference-level Vina Scores with comparable molecular size

因此 MolCRAFT 的主要 Vina 比较至少包含 controlled-size 设计。Pocket2Mol、TargetDiff、DecompDiff 原始主评测代码中没有核对到统一的 size-matched Vina 或 ligand-efficiency 主指标。

所以以下比较不能直接等同：

- MolCRAFT controlled-size Vina vs 其他方法无条件 pooled Vina；
- raw Vina vs ligand efficiency；
- 平均 heavy atoms 描述 vs 真正 size matching。

## 8. Reference ligand

TargetDiff summary notebook 加载每个 pocket 一个 reference ligand，并计算：

```text
generated docking affinity <= reference docking affinity
```

这得到的是 reference-level high-affinity 比例，不是生成分子平均 Vina。

DecompDiff README 也列出 Reference 行，但需要区分：

- reference ligand docking score；
- generated molecule mean docking score；
- high-affinity fraction；
- success rate。

四者不是同一统计量。

## 可比性总表

| 方法 | test pocket | 每 pocket 样本 | receptor | docking | exhaustiveness | 与 clean evaluator |
|---|---:|---:|---|---|---:|---|
| Pocket2Mol | 100 | 实际 `samples.finished` | protein_root PDB | QVina dock | 16 | 部分可比 |
| TargetDiff | 100 | `index[:100]` | `data/test_set` | score/min/dock 多模式 | 16 或 32，取决 evaluator | 部分可比 |
| DecompDiff | 100 | `index[:100]` | `./data/test_set` | `vina_full` 默认三阶段 | 32 | 部分可比 |
| MolCRAFT | 论文 CrossDocked2020 | controlled-size，但公开代码未完整核对 | 未完全核实 | 论文 Vina Score，模式未完全核实 | 未核实 | 不可直接等同 |

## 对 DFE-Unified clean evaluator 的建议

只有下列项目全部一致时，才允许声称严格可比：

1. 同一 CrossDocked 版本；
2. 同一 `split_by_name.pt` 与 100 test pockets；
3. 每 pocket 固定生成数量与有效分母；
4. 同一 receptor 文件；
5. 同一 box 中心/大小；
6. 同一 Vina 实现；
7. 同一 score-only/minimize/dock 模式；
8. 同一 exhaustiveness/num_modes；
9. 同一 reconstruction/multi-fragment 过滤；
10. 同一 pooled 或 pocket-macro 聚合方式。

建议同时报告：pooled mean、pocket macro mean、median、reference high-affinity、heavy atoms、ligand efficiency 和 direct PoseBusters。PoseBusters 应作为独立附加指标，不与 Success Rate 混写。
