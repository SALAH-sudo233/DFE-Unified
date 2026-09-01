# DFE-Unified 生成优先实验设计

## 1. 决策与研究边界

本设计以现有 DF 500K checkpoint 为固定诊断锚点，不重新训练或覆盖该模型。研究目标是先解释其在开放式口袋上的退化和 SE(3) 行为，再验证 BIF 与亲和力筛选，最后决定是否构建会反向影响生成器的 Unified 版本。

已冻结的产品优先级是“生成优先”：Pocket2Mol/DF 是主链路，BIF 和亲和力预测首先作为可失败、可旁路的筛选链路。Unified-v1 的生成候选必须与 DF 500K 原始候选逐条相同；筛选模块失败时返回未筛选候选，不阻断生成。

现有证据只支持以下事实：

- DF checkpoint 的 iteration 为 500,000，SHA-256 为 `34b2e8cadb7351c884e36cbfdee23de2a6ac2cb6fdd213612e401fd36c9d1fc0`。
- 已保留的评估只完成 30 个请求口袋中的 21 个，共 2,331 条后重构评估记录。
- 当前 DF 把包含 3 个绝对方向分量的 8 维特征送入普通 MLP，再将输出加到标量通道；严格旋转等变性没有得到保证。
- ADF/BIF 400K checkpoint 中 ADF、BIF 和 affinity 参数组没有优化器状态，不能作为有效训练证据。
- “开放口袋更差、闭合口袋更好”当前是待验证观察，不是已建立的统计结论。

不在本轮计划中声称真实结合、临床价值、物理力场、SOTA、完整 30-pocket 结果或已实现 Unified。

## 2. 候选路线与选择

### 路线 A：诊断优先、生成优先（采用）

先冻结口袋开放度与评估协议，用现有 DF 500K 做 SE(3) 单元/层级测试、推理干预和生成轨迹诊断。随后独立验证 BIF 与 ScreeningHead，Unified-v1 只做生成后筛选。只有通过独立 gate 的模块才能进入生成反馈或新一轮长训练。

优点是每一步都能被证伪、最大限度复用现有 checkpoint，并适配 0–4 张不稳定 GPU。代价是端到端 Unified 出现较晚。

### 路线 B：架构优先并行训练（不采用）

同时训练 DF-v2、BIF、DF+BIF 和联合 affinity 模型。四卡时墙钟时间较短，但当前等变性、开放口袋定义、BIF 梯度和评估分母均未解决，失败后无法归因。

### 路线 C：筛选器优先（暂缓）

将 DF 500K 冻结为候选生成器，集中优化 affinity 排序。这可以较快形成可用系统，但不能回答 DF/BIF 对生成机制的贡献。该路线只在 DF-v2 研究 no-go、而筛选器独立通过 gate 时作为条件交付。

## 3. 研究问题与可证伪假设

### RQ0：现有评估能否支撑开放/闭合口袋结论？

H0.1：口袋开放度与 DF 500K 的端到端有效率、PoseBusters、严重 clash、早停和分子尺寸无稳定关系。

H0.2：任何表面相关性可以由 pocket 体积、参考配体重原子数、蛋白家族或缺失的 9 个口袋解释。

只有在完整冻结集合、全尝试分母和 pocket-level 统计下拒绝这些零假设，才保留“开放度调节 DF 效果”的结论。

### RQ1：现有 DF 路径是否满足 SE(3) 契约？

H1.1：原始几何量满足平移不变、距离标量旋转不变和方向向量旋转等变。

H1.2：`AnalyticalDirectionField.field_proj` 的输出以及下游标量 logits 在随机旋转后保持不变。

若 H1.1 通过而 H1.2 失败，则问题定位为方向分量到标量 MLP 的表示契约，而不是 Pocket2Mol 整体等变模块。

### RQ2：DF 500K 是否真实利用 DF，开放口袋退化发生在哪一步？

通过 `normal`、`zero`、`direction-zero`、`node-shuffle`、`wrong-pocket` 和强度扫描进行推理期干预。推理干预只能回答 checkpoint 依赖性，不能替代独立训练的 no-DF baseline。

阶段诊断覆盖 initial frontier/focal、position GMM、element、bond/triangle attention、termination、reconstruction 和 pose checks。若开放度主要影响首原子或 frontier，则优先改进几何与置信度；若只影响 element/bond，则优先检查 BIF/化学头；若只在重构后出现，则先修评估管线。

### RQ3：BIF 是否提供可审计、非容量或配对伪影的化学信号？

BIF 定义为 Biochemical Interaction Field：在空间查询点上聚合口袋原子的电荷互补、供受体互补、疏水、芳香和位阻通道。解析通道与 learned projection 必须分开记录。BIF 不是 docking score、bond decoder 或 affinity head 的同义词。

独立有效性要求：

- 所有预期参数收到有限梯度、发生参数更新并产生优化器状态；
- 可以过拟合一个 8–32 复合物的小数据集；
- 在冻结的 protein-family 与 scaffold 隔离集合上，native contact/atom-role probe 优于 random、node-shuffled、wrong-pocket 和容量匹配控制；
- 可视化通道与由固定工具产生的氢键、盐桥、疏水和芳香接触标签有超出基准率的对应，而不是仅展示好看的热图。

### RQ4：筛选器是否能在不改变生成的前提下富集更好的候选？

ScreeningHead 是独立 learned affinity/ranking 模块，可使用 BIF、protein-ligand pairwise features 和 cross-attention。它必须在无泄漏 split 上独立通过回归、排序、校准和 pocket-shuffle 控制，之后才能用于 DF 500K 候选重排。

Unified-v1 采用约束排序：先保留通过预注册化学/pose gate 的候选，再按预测 affinity 排序；不把 QED、Vina 和 affinity 任意加权为一个不可解释总分。Vina 仅作离线评价 oracle，不计入在线筛选延迟。

### RQ5：何时允许 BIF/ScreeningHead 影响生成？

只有 Unified-v1 在盲测候选池上稳定富集，且满足生成非退化和效率门槛后，才进入 Unified-v2。优先级依次为：step-level 候选重排、受限 rejection、`BIF -> element/bond` 单向注入；不首先采用双向联合梯度训练。

## 4. 口袋开放度的可重复定义

开放度以连续变量为主，开放/中间/闭合标签只用于图表。对冻结 pocket center 发射 2,048 条 Fibonacci 球面射线；若射线在半径 12 Angstrom 内与任一蛋白重原子的范德华球相交，则记为被遮挡：

```text
enclosure = blocked_rays / 2048
openness = 1 - enclosure
```

坐标平移或旋转不改变该值。主分析使用连续 `openness`。展示用三分组阈值根据完整测试 manifest 的 openness 三分位数在看模型结果前冻结。

辅助协变量包括 pocket 重原子数、12 Angstrom 内原子密度、到最近蛋白原子的距离、参考配体重原子数、参考配体 SASA 暴露比例、pocket 体积和蛋白家族。参考配体只用于离线分层验证，不输入生成器。至少对 20 个口袋做双人盲标开放/闭合，报告与连续指标的一致性；人工标签不替代主指标。

## 5. SE(3) 验证契约

对至少 20 个真实 pocket/partial-ligand 状态，各采样 100 个 Haar 随机 SO(3) 旋转和 10 个随机平移。使用 `eval()`、固定 dtype 和耦合随机数。

| 输出 | 预期变换 |
| --- | --- |
| 距离、场强、sigma、mixture weight、frontier/element/bond logits | 不变 |
| 原始方向、VN/GVP 向量、position mean、生成坐标 | 随旋转协变并随平移移动 |
| loss、终止概率、排序分数 | 不变 |

float32 确定性前向的主要 gate 为归一化最大误差小于 `1e-4`，float64 解析 DF gate 为 `1e-8`。超过阈值必须输出首个失败模块、最大/中位/p95 误差和输入 hash。反射属于 E(3) 附加诊断，不作为 SE(3) 主 gate。

对于离散采样轨迹，先比较共享前缀上的 logits 和连续参数；使用相同基础随机数比较坐标。分支一旦不同，记录首个分叉步骤，不把后续不同长度轨迹直接做坐标均方误差。

## 6. Phase 0：现有 DF 500K 诊断

### P0-A 评估分母与 manifest

冻结全部待测 pocket、protein、center、checkpoint、代码、环境和 seed。每次生成尝试从开始即写入 ledger，包括 sampling、reconstruction、SDF、docking 和 pose-check 状态。主要分母是全部尝试，不再从成功 SDF 开始计数。

保留现有 21-pocket 结果作为历史证据，新的诊断运行不得与之覆盖或合并成同一个 run。

### P0-B 低成本机制测试

无需训练地完成开放度计算、SE(3) 层级测试和现有 checkpoint 推理干预。先在 6 个口袋（开放/中间/闭合各 2）做 `10 attempts x 1 seed` smoke；通过后在冻结 30-pocket 集合做 `20 attempts x 3 seeds`。若资源允许，再扩展至完整官方 test split。

干预矩阵：

| ID | 干预 | 解释边界 |
| --- | --- | --- |
| D0 | normal DF | 固定锚点 |
| D1 | DF output gate = 0 | checkpoint 对 DF 总输出依赖 |
| D2 | 三个方向输入置零，保留 5 个标量 | 方向分量依赖 |
| D3 | pocket 内 node-wise DF shuffle | 空间配准依赖 |
| D4 | center-aligned wrong-pocket DF | pocket 特异性依赖 |
| D5 | gate = 0.25/0.5/1.0/1.5 | 强度与开放度交互 |

D1 不是公平 no-DF 模型，D2–D5 也不是训练消融；结论必须使用“推理干预”。

### P0-C 失败阶段定位

每个自回归 step 记录 frontier 数量与熵、focal、position GMM 参数、DF 幅值与方向、候选到 pocket 的距离、element/bond logits、价态 mask、终止原因和耗时。派生指标包括首原子成功率、每步 pocket containment、DF-position alignment、过早终止、超出 pocket、clash 首发步骤、重构失败和断连率。

主统计使用 pocket 为聚类单位的 hierarchical bootstrap。对连续 openness 拟合 `metric ~ intervention * openness + pocket_size + ligand_size`，以 pocket bootstrap 给出交互项置信区间。三分组只用于可视化，不替代连续分析。探索指标进行 Benjamini-Hochberg FDR 校正。

### Phase 0 出口

只有同时得到以下产物才进入新架构训练：

- 可复算 openness manifest 和标签一致性报告；
- SE(3) 首个失败位置及误差谱；
- 至少一个能区分开放与闭合退化机制的 stage-level 指标，或明确报告未找到机制；
- 干预对照的 pocket-level effect 与置信区间；
- 端到端全尝试分母。

## 7. DF-v2 候选与选择规则

Phase 0 前不预先宣布主架构。候选按最小风险排序：

1. `DF-invariant`：仅将距离、场强和由点积/范数构成的不变量送入标量通道。
2. `DF-vector`：标量特征进入标量通道，方向进入现有 VN/GVP 向量通道；这是严格保留方向信息的推荐候选。
3. `DF-multiscale-confidence`：用多尺度径向核聚合方向，并输出 concentration/confidence；开放或低置信区域通过 gate 回退到 Pocket2Mol 表示。

不得把原始 xyz 方向分量直接经普通 MLP 声称为标量等变特征。每个候选先通过解析与模型级 SE(3) gate、梯度/优化器 gate、tiny-overfit 和 20–50K 短训。短训最多保留两个候选进入相同预算正式训练。

公平 no-DF baseline 必须来自原始 Pocket2Mol checkpoint 或与候选同数据、优化器、步数和 checkpoint 选择规则训练的模型。容量匹配 control 用与 DF projection 参数量相当、但不接收 DF 的 invariant MLP。

## 8. BIF 独立验证

### 8.1 BIF 通道

第一版只使用可审计通道：steric/VDW occupancy、electrostatic complementarity、H-bond donor、H-bond acceptor、hydrophobic 和 aromatic。每个通道的原子类型规则、单位、核宽和截断距离写入版本化 schema。若部分电荷来自近似规则，明确称为 proxy，不称为物理电势。

### 8.2 单元与训练 gate

在正式训练前必须自动证明：

- 平移/旋转不变通道保持不变，若输出向量则按 SE(3) 协变；
- mask、空 pocket、重叠点和极远点产生有限输出；
- BIF、projection 和 affinity 参数均有非零有限梯度；
- optimizer step 后参数发生变化且 checkpoint 中每个预期参数有 optimizer state；
- 标签置乱后性能回到 chance/基线，wrong-pocket 和 node-shuffle 显著破坏匹配信号。

### 8.3 独立任务

任务 B1 是 native ligand atom role/contact probe：在真实配体原子和距离匹配 decoy 点上预测元素角色与 PLIP/固定几何规则接触标签。报告每通道 AUROC、AUPRC、top-k enrichment、校准和 protein-family 分层结果。

任务 B2 是 BIF-assisted element/bond probe：冻结 Pocket2Mol context，比较 no-BIF、BIF、random-BIF、wrong-pocket 和容量匹配 control。只有真实 BIF 在 held-out protein family 上改善 element macro-F1、bond macro-F1 或 molecule-level topology proxy，且控制组不产生同等收益，才认为 BIF 提供独立信息。

## 9. ScreeningHead 与无泄漏数据

训练/验证/测试按 protein sequence cluster 与 Bemis-Murcko scaffold 的连接分量分组，使同一 protein family 或 scaffold 不跨 split。具体数据版本、聚类阈值、标签单位和所有条目 hash 在训练前冻结。随机 split 只能作为诊断，不能作为主结果。

必须比较 ligand-only、pocket-only、简单 descriptor、无 BIF、BIF、无 pairwise、无 cross-attention、full model 和 label-permutation。主指标为 MAE/RMSE、Pearson/Spearman；有同靶点排序集时报告 per-target Spearman、EF1%、EF5% 和 BEDROC。回归不确定性报告 50/80/95% 区间覆盖与校准误差。

筛选 hard gate：

- protein/scaffold 隔离测试集 Spearman 至少 `0.40`；
- MAE 不劣于冻结的最佳简单 baseline，且 pocket-aware full model 相对 ligand-only 的 paired bootstrap 95% CI 不跨 0；
- pocket shuffle 明显破坏结果，label permutation 回到基线；
- 在可用排序集上 enrichment 的 pocket-bootstrap 95% CI 下界大于随机水平；
- BIF 可解释通道的接触定位优于 prevalence 和 shuffled BIF。

若未通过，ScreeningHead 只能作为实验模块，禁止使用“affinity-aware Unified”表述。

## 10. Unified-v1：生成后约束筛选

对每个 pocket/seed 由 DF 500K 一次生成固定候选池，所有 ranker 复用完全相同的候选和失败记录：

- random rank；
- QED/简单 descriptor baseline；
- rule-only BIF compatibility；
- ScreeningHead affinity；
- constrained BIF + ScreeningHead；
- Vina rank 仅作离线 oracle 上界。

约束 ranker 先使用冻结的 sanitize、价态、严重 clash 和最小多样性规则建立 eligible set，再按 affinity 排序；若 eligible 数量不足，按记录明确回退，不静默补充。报告 top-1/top-5/top-10、全候选分布和 selection coverage。

生成非退化以 DF 500K 为基准：由于 Unified-v1 不改变生成，raw candidate 的 validity、PoseBusters 和 diversity 应逐条相同。系统级验收同时要求：

- 端到端 validity 下降不超过 2 个百分点；
- PoseBusters pass rate 下降不超过 3 个百分点；
- pocket-level diversity 下降不超过 `0.03`；
- 上述指标无 FDR 校正后的显著退化；
- DF 生成加 BIF/ScreeningHead 在线推理的每个有效分子时间增幅不超过 15%；Vina 离线评价时间不计入在线延迟；
- selected top-k 的盲测 affinity/docking 或 interaction-compatible 指标优于 random rank，且不是以更小分子、单一 scaffold 或低 diversity 换取。

## 11. Unified-v2：受控生成反馈

仅在 Phase 0、BIF、ScreeningHead 和 Unified-v1 全部过 gate 后启动。按风险从低到高测试：

1. 每步 element/bond 候选的 BIF 重排，不更新生成器；
2. 固定最大额外候选数的 rejection/rerank；
3. stop-gradient `BIF -> element/bond` feature injection；
4. 仅在前三者失败且资源充足时测试 joint-training baseline。

每个反馈臂都必须与 DF-only、random/shuffled BIF 和容量匹配 control 比较，并满足相同生成非退化门槛。双向 `G <-> S` 注入不属于 Unified-v2 的首轮范围。

## 12. 统计、复现与指标纪律

实验单位是 pocket，不是分子。先在 `pocket x seed` 内聚合，再做 paired pocket-level 差值。主要结果报告 effect size、median/IQR、10,000 次 cluster bootstrap 95% CI 和 seed 方差；显著性检验使用 paired permutation 或 Wilcoxon，并对预注册主要指标做 BH-FDR 5%。

所有表必须同时报告 attempts、generated、reconstructed、valid、dockable 和 checked 数量。缺失 docking/pose 不做均值填补；失败留在端到端分母。模型选择不查看最终 test；开放度阈值、primary endpoints、随机种子和停止规则均在运行前写入 run manifest。

每个 run 绑定 Git commit、dirty state、config hash、dataset/split hash、checkpoint hash、容器/环境、GPU、seed、开始结束时间和父 run。原始输出不可覆盖，汇总表只由脚本生成。

## 13. 0–4 GPU 调度与停止规则

- `0 GPU`：开放度 manifest、数据/泄漏审计、CPU 指标测试、统计、BIF 标签构建、结果汇总。
- `1 GPU`：SE(3) 模型测试、DF 500K trace、单臂 smoke/短训、ScreeningHead 单 seed 开发。
- `2 GPU`：同一配置两个 seed，或一个 baseline 与一个候选；不在确认前扩展新臂。
- `3 GPU`：一个配置三个预注册 seed，是正式稳定性实验的优先用法。
- `4 GPU`：三个 seed加一个 baseline/候选 smoke，或四个独立诊断臂；除非代码已有验证，不做脆弱的多卡同步训练。

所有长任务按固定间隔原子化 checkpoint，可从最后一个完整 checkpoint 恢复。出现 NaN、梯度缺失、optimizer-state 缺失、数据泄漏、SE(3) gate 失败、三次连续验证恶化或生成非退化门槛失败时停止该臂，不用额外算力掩盖机制错误。

## 14. 阶段性交付与 Go/No-Go

| Gate | Go | No-Go 后处理 |
| --- | --- | --- |
| Phase 0 | 开放度、SE(3)、干预和 stage trace 均可复算 | 保留 DF 500K 为历史模型，先修管线/契约 |
| DF-v2 | SE(3) 通过且短训跨 seed 显示开放口袋非退化 | 不跑 500K，回到 invariant fallback |
| BIF | 梯度/优化器、probe、shuffle/wrong-pocket 控制通过 | 不注入生成器，仅保留规则分析 |
| Screening | 无泄漏预测、校准、富集通过 hard gate | Unified 降级为生成器加规则筛选 |
| Unified-v1 | 富集提升、生成不退化、在线开销 <=15% | 保留 DF 500K 单模型与独立筛选报告 |
| Unified-v2 | 反馈改善至少一个预注册生成/化学指标且其他不退化 | 不做双向/联合长训 |

最终可能有三个诚实交付：生成方法版、独立筛选版、或通过全部 gate 的 Unified 版。阶段结果决定叙事，不能反向调整实验门槛迎合预期。
