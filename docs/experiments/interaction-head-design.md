# DFE-Unified 低成本 protein-ligand interaction head 设计

## 结论

当前生成 encoder/field 已经有 pairwise 距离、相对向量、compose kNN 图和 ligand context bond type，但这些信息服务于生成任务，未被显式约束为 pair-specific affinity interaction。建议新增一个独立 interaction head，不改生成 head 的输出语义，不直接把 affinity loss 加入 frozen checkpoint。

## 可复用与需新增

| 信息 | 当前状态 | 复用/新增 |
|---|---|---|
| protein-ligand 距离 | `compose_pos`、kNN edge、`distance_expansion`、`vec_ij` 已存在 | 直接复用；head 输入需区分 protein/ligand 节点 |
| 相对方向 | `vec_ij=pos_query-pos_compose`，VN vector channel 已存在 | 复用其方向不变量：距离、单位向量与 dot/cross 派生标量；不直接破坏 SE(3) |
| 元素类型 | protein 5 元素 one-hot；ligand 7 元素 one-hot | 直接复用 `protein_atom_feature`/`ligand_atom_feature_full` 的元素片段 |
| 芳香性 | 当前 transform 的 ligand feature 不是显式芳香性字段；SDF/RDKit 可计算 | 需新增 atom-level aromatic flag，且固定 kekulize/芳香性处理协议 |
| HBD/HBA | 当前 transform 没有显式 HBD/HBA | 需用 RDKit 在原始配体上计算并保存；protein 端可按 atom/residue donor/acceptor 规则编码 |
| 局部接触图 | `compose_knn_edge_index`、`compose_knn_edge_feature` 和 bond type 已存在 | 直接复用；新增 protein-ligand contact mask、pair distance bins |
| 真实亲和力标签 | PDBBind refined 有 pKd/pKi，但 CrossDocked 诊断样本不是同一标签协议 | 需新增 label manifest；训练/验证按 protein sequence cluster 隔离 |

## 最小 head

对每个 complex，取 protein 与完整 ligand 的 atom/node 表征，构造半径 5 Å 或 6 Å 的 protein-ligand contact pairs。对 pair `(p,l)` 输入：

```text
z_pl = MLP([
  h_p_scalar, h_l_scalar,
  RBF(||r_p-r_l||),
  onehot(element_p), onehot(element_l),
  aromatic_p, aromatic_l,
  donor_p, acceptor_p, donor_l, acceptor_l,
  contact_mask,
  optional invariant dot(vector_p, vector_l)
])
```

对接触 pair 做 attention/log-sum-exp pooling，再与 ligand-only 和 pocket-only pooled features 拼接：

```text
z_pair = logsumexp(pair_mlp(pair_features), dim=pair)
z = concat(z_pair, mean(h_ligand), mean(h_protein), log1p(n_ligand), log1p(n_protein))
ŷ = MLP(z)  # scalar affinity, preferably z-scored pKd/pKi
```

第一版应只使用 scalar invariant features 和显式 pair geometry；不要把一个高维 vector pooling 直接压成 affinity，避免把方向信息错误地当作坐标系相关量。

## 正负样本

首选 PDBBind refined 的真实 complex 作为 positive，label 为统一单位的 pKd/pKi（先固定 Kd/Ki 合并规则并在 manifest 中记录）。negative 不应随机任意配对，因为会让 protein/ligand 边际先验足以解题。采用三层负样本：

1. **Pocket shuffle**：在 batch 内打乱 ligand 与 pocket 配对，保持 ligand、pocket、heavy atoms 和标签边际分布。
2. **Geometry shuffle**：保持 protein/ligand atom types 与距离分布近似不变，随机置换 ligand 坐标或局部 contact assignment。
3. **Hard negative**：同一 protein family 内匹配相近 MW/HA、相似 ligand descriptor 但不同 pocket/complex。

loss 使用 pairwise ranking + regression 的轻量组合：

```text
L_aff = Huber(y_pred, y_true) + 0.25 * BCEWithLogits(s_pos - s_neg, 1)
L_total = L_generation + lambda_aff * L_aff
```

第一轮建议 `lambda_aff=0.02`，并在 `0.005/0.02/0.05` 做小矩阵；训练初期先 warm-up 1k steps 只训练 head，再逐渐允许最后 1–2 个 encoder block 更新。

## 严格对照

| 对照 | 保留 | 打乱/移除 | 目的 |
|---|---|---|---|
| full pair head | protein、ligand、contact geometry | 无 | 主实验 |
| ligand-only | ligand 表征、HA、QED 等 | 所有 protein/pair | 测 ligand 边际先验 |
| pocket-only | protein 表征与 pocket geometry | 所有 ligand/pair | 测 pocket 边际先验 |
| pocket-shuffle | 原始 node features | ligand-pocket 配对 | 测 pair-specific 信号 |
| size-matched | pair/ligand/pocket | negative 按 HA/MW 匹配 | 排除尺寸捷径 |
| geometry-shuffled | 元素/计数 | 相对坐标/contact | 测几何交互依赖 |
| label permutation | 全部输入 | 训练标签置换 | 检查泄漏/代码错误 |

报告 test Pearson/Spearman、RMSE、pairwise accuracy，并单独报告 full − max(ligand-only,pocket-only,pocket-shuffle)；不能只报 full Pearson。

## 代码级最小改动

只在独立 feature branch 或独立目录实现：

- 新增 `models/interaction_head.py`：pair feature encoder、contact pooling、regression/ranking head。
- 新增 `utils/interaction_features.py`：RDKit donor/acceptor/aromatic flags、元素和距离特征。
- 新增 `utils/interaction_transforms.py`：从 `ProteinLigandData` 构造 pair index、contact mask、负样本索引。
- 新增 `configs/interaction_probe.yml`：label、split、lambda、freeze、seed。
- 在独立 probe runner 中调用现有 `embed_compose`/encoder；不改 `MaskFillModelVN.get_loss`，先进行 frozen representation probe。
- 只有 frozen probe 明确超过三种边际对照后，才新增 joint loss hook。

可直接实现：距离/RBF、相对向量派生不变量、元素、现有 kNN/contact 图、ligand/pocket pooling、shuffle controls。需要新增数据：可靠 HBD/HBA/aromatic atom flags、complex affinity label manifest、sequence-cluster split、hard-negative matching。

## 最小实验矩阵

第一阶段只做 frozen probe：

```text
representation: full pair / ligand-only / pocket-only / pocket-shuffle / geometry-shuffled
label: pKd/pKi z-score
split: sequence-cluster train/val/test
seeds: 20260914, 20260915, 20260916
```

第二阶段只在 full pair frozen probe 有稳定增益后做短 joint fine-tune：

```text
lambda_aff: 0.005, 0.02, 0.05
trainable: head only; head + last encoder block
replay: generation loss on every batch
```

## 失败判据

- label permutation test Pearson 不接近 0，或 pocket-shuffle 仍接近 full：停止，优先查泄漏/边际捷径。
- full pair 在三个 seed 上相对最强边际 baseline 没有稳定增益：停止，不加 joint loss。
- interaction head 使 generation validation loss 上升超过预先设定容忍度，或 fixed-prompt generation 的 validity/PB/HA 分布显著漂移：停止 joint fine-tune。
- 只提升 raw affinity/Vina，但 size-matched 或 ligand-efficiency 指标不提升：判为尺寸捷径。
- family-held-out test 退化，即使随机 split 提升：不宣称泛化。

## 资源估计

frozen probe 只需缓存 node embedding 和 pair index，显存主要由现有 encoder 推理决定；相对 full generation training 只增加一个小 MLP 与 contact pooling，预期参数量为几十万量级。训练 head-only 的显存增量很小；最后一个 encoder block 解冻会增加反传激活，建议 batch size 4–8、gradient accumulation 2–4。这里是设计估计，不是实测资源结果。
