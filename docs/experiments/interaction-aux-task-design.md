# 最小 protein–ligand interaction-aware 辅助任务设计

- **状态**：仅设计与代码审查；未修改 `repo/main`、未覆盖 checkpoint、未启动训练或 GPU 作业。
- **审查对象**：`models/encoders/cftfm.py`、`models/df_module.py`、`models/fields/classifier.py`、`models/fields/interaction_residual.py`、`models/maskfill.py`、`utils/transforms.py`。
- **目标**：以最小改动验证共享的蛋白–配体局部相互作用表示是否真的包含 pair-specific signal，而不是只重复 ligand/pocket 边际信息。

## 1. 当前实现审查

### 1.1 Encoder

`CFTransformerEncoderVN`（默认 6 个 `AttentionInteractionBlockVN`）接收节点的 scalar/vector 表示、KNN/半径图边和边特征；每层使用相对位移的距离 Gaussian expansion 与向量 expansion，消息按接收节点聚合，并保持 scalar/vector 两路。输出尺寸由 `encoder.out_sca/out_vec` 暴露。该表示是最适合复用的 backbone 特征：

- `h_compose[0]`: 每个 compose 节点的 invariant scalar hidden，包含蛋白和已生成配体上下文；
- `h_compose[1]`: 每个节点的 equivariant vector hidden；只在构造 pair 特征时使用旋转不变量（范数/点积），不把绝对坐标输入 head；
- edge distance/edge type：可复用现有图和相对距离，但不复制一套 encoder。

`maskfill.py:199–215` 的实际顺序是 embed → encoder → 可选 DF scalar 加和，因此辅助头应接在 **DF 加和之后的最终 `h_compose`**，避免训练/推理表征错位。需要在训练 forward 处保留同一批次的节点索引和坐标。

### 1.2 Direction Field（field）

`AnalyticalDirectionField.raw_features` 为查询点计算 8 个确定性输入：最近蛋白距离、最近蛋白方向向量、电势、疏水加权距离、距离平方、逆距离（实际拼接为 8 维）；再由 `field_proj` 投影到 `df_dim`，并通过 `df_proj` 加到 encoder scalar 通道。它是强的几何/物性先验，但本身不含 ligand atom identity，也不建模一个具体 protein atom–ligand atom pair。因此：

- 作为辅助 head 的可选输入只取 **已投影后的最终节点表示**，不要把 field raw 值再次拼进去；
- 不把 DF 单独当作 interaction evidence；需证明打乱 protein 条件会破坏 pair 预测；
- 若 `df_dim=0`，设计仍须能运行。

### 1.3 Classifier / edge path

`SpatialClassifierVN` 先从 compose 节点向 query 点消息聚合，得到 `y`，再做元素分类；边预测用 query 节点 `y_i`、compose 节点 `h_j`、相对距离/向量，经 `edge_feat`、`AttentionEdges` 和 `edge_pred` 预测 `num_bond_types+1` 类。`AttentionEdges` 的 triangle bias 来自 `tri_edge_feat` 与 triangle 相对距离；当前 `utils/transforms.py:590–598` 产生的是 edge type `[-1,0,1,2,3]` 的 one-hot。

现有 `InteractionResidual` 已在 `AttentionEdges` 中预留，但 `AttentionEdges.__init__` 的默认 `interaction_dim=0`，`get_field_vn` 也未见将其打开；即使打开，`classifier.py:189` 只取 `tri_edge_feat[:, :interaction_dim]`，这些目前主要是离散 triangle edge-type 特征，不是明确的 protein–ligand pair supervision。并且 residual 零初始化，适合 checkpoint-safe 增量，但不应把它误称为已训练 interaction head。

### 1.4 Transforms / labels

现有 transform 提供 protein/ligand 原子 one-hot 与拓扑计数、H 删除、ligand mask/BFS mask、compose KNN/triangle 索引等。没有现成的 protein–ligand contact label。辅助任务因此应在 batch 内从真实复合物坐标与 mask **确定性构造**，而非使用 docking 后分数或生成结果标签。

**mask 可见性约束**：当前 `AtomComposer` 的 `compose_pos` 是 `ligand_context_pos + protein_pos`（`utils/transforms.py:637–654`），因此默认训练路径的 `h_compose` 只包含 context ligand。首轮辅助任务必须二选一并预注册：

- 推荐：标签 pair 也只在 `ligand_context` 与 protein 之间构造；被 mask 的 ligand 原子完全不进入 pair 特征或 pair 标签；
- 或者：新增独立、未 mask 的评估/预训练 transform，显式构造全 ligand compose，并保证该输入分布与目标一致。

禁止用 mask 后不可见原子的真实位置构造标签，再从 context 表示预测它；这会把“接触分类”混成泄漏的重建任务。每个 pair 必须携带 `complex_id`、protein/ligand 原始索引和 compose 索引，batch collate 后显式加 offset；不得依赖 PyG 对自定义 pair 字段的隐式 offset。

## 2. 推荐的最小辅助任务

### 2.1 任务定义：masked pair-contact discrimination

对每个真实复合物，取 ligand heavy atoms 与 pocket protein heavy atoms 的候选 pair：

- 正样本：真实坐标下 `d_ij <= 4.5 Å` 的 protein–ligand pair；
- 负样本：同一复合物内 `d_ij >= 6.0 Å` 的 pair；从中按正样本数 1:1 采样，距离区间 `[4.5,6.0)` 丢弃，减少边界噪声；
- 若正样本过多，最多每个复合物 `K_pos=64`，按距离优先/固定 seed 抽样；负样本同数；
- pair 必须来自同一个 complex，禁止跨 pocket 配对；按 complex/pocket 划分 train/valid/test，禁止同 pocket 泄漏；
- 仅 heavy atoms，且 protein 原子须在有效 pocket mask 中；不使用 H（主 transform 已删除 H）。

标签是 `z_ij∈{0,1}`，语义为“近接触”，不是氢键、共价键或 binding affinity。若数据没有可靠共晶 ligand pose，该样本跳过并计数；不能用预测 pose 伪造正样本。

### 2.2 可复用 pair 特征

对 protein 节点 `p` 与 ligand 节点 `l`，从最终 `h_compose=(h_s,h_v)` 构造：

```text
u = h_s[p] || h_s[l]                         # 2*S
r = ||h_v[p]|| || ||h_v[l]||                 # 2*V
c = sum_k h_v[p,k] * h_v[l,k]                # V，旋转不变量
q = ||x_p - x_l||, Gaussian(q; 32 bins)      # 32
chem = protein_atom_feature[p] || ligand_atom_feature[l]
```

推荐最小版本先只使用 `u || r || c || Gaussian(q)`；chem 作为第二个预注册变体，不能在看结果后选择。相对向量只以距离进入 head；若需要方向信息，使用 `dot(h_v[p,k], Δx)` 或其归一化形式，禁止绝对坐标。pair head 为共享 MLP：`Linear(D,128) → SiLU → Dropout(0.1) → Linear(128,1)`，参数约 `128*D+...`，远小于 encoder。为保证类别公平，logit 取对称 pair 表示（本任务 protein→ligand 有方向，但可额外报告交换 sanity check）。

### 2.3 Head 与输出

新增独立 `InteractionAuxHead`（建议目录 `models/auxiliary/interaction.py`）：

```python
logit_ij = head(pair_features)[:, 0]
return logit_ij, pair_index, pair_label, pair_distance
```

Head 不复用 `SpatialClassifierVN.classifier`，因为后者输出 query element class，且输入是聚合后的 query 表示；也不直接改写 `AttentionEdges`。第一阶段只读 `h_compose`，不把 aux logit 回注主模型。第二阶段若任务有效，才把同一 pair MLP 的低维输出映射为 `interaction_features`，接入已存在的零初始化 `InteractionResidual`；这避免同时改变目标和生成路径。

## 3. Loss 与组合权重

主任务 loss 保持现有实现完全不变（frontier、position、element、bond 等）。辅助 loss：

```text
L_aux = BCEWithLogits(pos_weight = N_neg/N_pos, logits, z)
L_total = L_main + λ_aux * L_aux
```

建议预注册 `λ_aux=0.05` 作为最小扰动；仅做 `0`（严格主模型对照）、`0.05`（主实验）、`0.2`（敏感性）三个值，不能以测试集调权重。对每个 complex 先均衡采样再 batch mean，避免大 pocket 支配 loss。记录 `L_main`、`L_aux`、正负数目和有效 complex 数。

### 防止 trivial shortcut

- 距离是必要信号，但需报告 **distance-only logistic baseline**；若 aux 仅与该 baseline 持平，不能称 interaction-aware；
- 同时做 protein feature shuffle（保持坐标与 pair index 不变）和 pocket shuffle（配对到另一 pocket，坐标按该 pocket 内构造），标签不变；性能应显著下降；
- label permutation 作为负控，性能应接近 chance；
- train/valid/test 按 pocket/complex 分组，不能随机按 pair 切分；
- 负样本不得来自仅靠远距离的不同分布而造成标签泄漏，主报告同时提供 distance-matched hard negative：在每个正样本附近的远距离候选中采样。

## 4. 严格对照矩阵

最小首轮只运行 CPU/代码级准备，不在本交付启动训练。正式实验的预注册矩阵：

| ID | backbone | aux 输入 | `λ_aux` | 用途 |
|---|---|---|---:|---|
| C0 | frozen/同训练设置 | 无 | 0 | 严格主模型基线 |
| C1 | 同 C0 | distance-only | 0.05 | 几何下限 |
| C2 | 同 C0 | `h_s` pair + distance | 0.05 | scalar pair 增益 |
| C3 | 同 C0 | `h_s+h_v` invariant + distance | 0.05 | 推荐主实验 |
| C4 | 同 C3 | protein feature shuffle | 0.05 | 条件依赖负控 |
| C5 | 同 C3 | pocket/label permutation | 0.05 | 泄漏/记忆负控 |

若 C3 相比 C1 无独立提升，停止接入 residual。若 C3 有效，第二阶段才比较：主模型、C3 aux-only、C3 + zero-init residual（同一预算）。不把 frontier 消融、A3 vector attention、A4 triangle bias 与本辅助任务混在首轮因果结论中。

## 5. 评估与失败判据

### 必报指标

- pair ROC-AUC、PR-AUC、average precision、按 pocket macro 平均；
- hard-negative AUC；distance-only 与 label-permutation 对照；
- calibration（Brier/ECE）；正负 pair 数、跳过样本数；
- 生成主任务（若进入第二阶段）：固定 pocket 与固定 seed/预算，报告 termination/productivity、heavy atoms 匹配后的 Vina/LE、QED、PoseBusters/碰撞/键几何和多样性；保存每个 pocket 原始分子与失败原因；
- pocket shuffle 应显著退化；标签置乱应接近 0.5 ROC-AUC。

### 预先定义的失败判据

1. C3 在独立 test pocket 上相对 C1 的 PR-AUC 提升小于 **0.03**，或 bootstrap 95% CI 跨 0：不称 interaction-aware，停止；
2. hard-negative AUC 不高于 0.55：模型只学距离/阈值，停止；
3. pocket shuffle 或 protein feature shuffle 退化小于 0.05 AUC：存在边际 shortcut/标签泄漏，停止并审计；
4. label permutation AUC > 0.55：数据切分、采样或实现错误，结果无效；
5. 第二阶段生成质量任一关键可信性指标（PoseBusters/碰撞/键几何）相对 C0 明显恶化，或尺寸未匹配时只改善原始 Vina：拒绝生成改进结论；
6. aux 导致主任务验证 loss/termination 在预注册容忍范围外恶化（建议主指标超过 C0 的 5%）：回退 `λ=0.05`/冻结 backbone，仍恶化则失败。

## 6. 代码改动清单（设计，不执行）

1. **新增** `models/auxiliary/interaction.py`：pair 特征、采样器、`InteractionAuxHead`、shape/device/assert；不改现有 checkpoint key。
2. **新增** `utils/interaction_labels.py`：heavy-atom pair 候选、4.5/6.0 Å 标签、hard negative、分组 split 检查；所有随机采样接受 seed。
3. **新增** `tests/test_interaction_aux.py`：标签边界、无正样本跳过、batch offset、旋转不变性、shuffle/label-permutation sanity、空 pair。
4. **新增** 独立训练入口（例如 `scripts/train_interaction_aux.py`），通过显式 flag 调用现有 model forward；不得改 `train.py`、`main` 或默认 config。
5. **仅接口性改动（第二阶段才做）**：让 `MaskFillModelVN` 以可选 hook 返回最终 `h_compose`；将 aux head 作为外部 wrapper，而非注册进默认模型；若接 `InteractionResidual`，显式传 `interaction_dim`，保持其末层零初始化和 `ablation` 开关。
6. **checkpoint**：C0 直接加载旧 checkpoint；aux wrapper 新增 key 存在时严格区分 `missing/unexpected`，禁止覆盖原文件；保存到独立输出目录。

本交付不实施以上代码改动，仅记录路径和接口。

## 7. 显存、计算与工程预算

设每个 complex 采样 `K_pos=K_neg=64`，pair 总数 `M≤128`，pair 输入维度约 `D=2S+3V+32`（默认 `S=256,V=64`，约 704）。

- head 激活约 `M*128`，约 0.07M 元素；即使 fp32 也远小于 encoder 的节点/边激活；
- pair gather 与 MLP 计算为 `O(MD)`，相对 encoder `O(E·hidden)` 可忽略；
- 不构造全 `P×L` dense 矩阵，先 radius/candidate 或分块计算；候选上限 `M` 是硬上限；
- 标签构造可在 CPU/offline manifest 完成，训练只传 pair index/label；
- 不增加向量通道、不复制 encoder、不保存 triangle 全图；预计显存增量在 batch 级为低个位数 MB，实际必须用 `torch.cuda.max_memory_allocated`（若未来允许 GPU）记录，而本任务不启动 GPU；
- CPU 设计检查只做 import/shape/unit test，不执行训练。

## 8. 结论

最小可检验路径是 **只读最终 compose 表征的 masked pair-contact BCE head**。它明确使用真实同复合物正负 pair、对 hard negative 和 distance-only baseline 有严格控制、可用 C0/C1/C2/C3/C4/C5 归因，并与现有零初始化 `InteractionResidual` 解耦。只有 pair head 在未见 pocket 上证明超越距离基线且通过 shuffle/置乱负控，才值得把共享 interaction 表征接入 triangle edge attention residual；否则不应增加主生成路径复杂度。
