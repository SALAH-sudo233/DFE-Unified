# Track B 消融结果与统一交互模块更新

## 证据范围

来源：`Pocket2Mol/outputs/fair_comparison_100k.json`。当前记录包含 DF-375k 完整模型与 A0/A3/A4 三个消融臂，均为 1a9q 单口袋记录；A1（去 frontier）和 A2（单高斯 position）不在该报告中。样本数为 22–28，且完整模型与消融的训练标记/预算需要从原始日志进一步核验。因此以下结果只用于**选择候选修改点**，不能作为完整因果排序或 CrossDocked-100 SOTA 结论。

| 模型 | Vina Dock 均值 | QED | MW | docking success | 多样性 |
|---|---:|---:|---:|---:|---:|
| DF-375k 完整 | -7.949 | 0.5198 | 294.38 | 0.8182 | 0.8177 |
| A0 去 DF | -7.274 | 0.5793 | 208.57 | 0.6667 | 0.7857 |
| A3 去向量注意力 | -6.986 | 0.4522 | 207.78 | 0.6087 | 0.8148 |
| A4 去三角 bias | -6.766 | 0.5420 | 203.19 | 0.3929 | 0.8510 |

尺寸差异很大，raw Vina/QED 不能脱离 heavy-atom 或 MW 匹配解释。`docking success` 的具体定义也必须与 PoseBusters 分开报告。

## 修改决策

第一优先修改 **A4 对应的键边注意力/triangle-bias 接口**，但不删除现有 triangle bias，也不同时重写 DF、vector attention 或 position head。加入蛋白条件化的 interaction residual：

```text
局部蛋白环境 + 候选/配体原子 + 相对几何/化学特征
                         ↓
                 interaction encoder
                  ↙                 ↘
       生成端：edge-attention residual   筛选端：pairwise pooling
```

原因：A4 结果提示该接口对可对接生成敏感；P3/P4 已显示仅池化 ligand/pocket 边际表示不足以表达配对特异性。该更新的目标不是追求更大的分子，而是同时改善：

1. **生成**：同尺寸 Vina Dock、终止率、有效产量和多样性；
2. **可信**：原始生成姿态的 PoseBusters/碰撞/键几何，而非只对接后分数；
3. **筛选**：固定候选池的 EF/BEDROC，并要求 pocket shuffle 显著退化。

新残差应零初始化，保证加载现有 checkpoint 时初始行为近似不变。方向特征只使用相对向量和旋转不变量，禁止绝对坐标 shortcut。

## 代码范围与验收

- 代码入口：`models/fields/classifier.py` 的 `SpatialClassifierVN`、`AttentionEdges`、`AttentionBias`。
- A3 的向量注意力 gate 仅作为消融开关保留；参数存在不等于功能存在，评测须记录实际 gate 状态。
- 不把 frontier 终止修复混入本次 interaction module 结论；frontier 是独立实验因素。
- 首轮必须有：原模型微调对照、interaction-residual 生成训练、共享 interaction 表示筛选训练。
- 所有结果按 pocket 保存原始分子、失败原因、heavy atoms、QED、Vina Dock、LE、PoseBusters 和 termination/productivity；不得只保存均值。
- 只有在尺寸匹配、pocket-shuffle、标签置乱和独立可信性指标均通过后，才允许声称 unified generation+screening 改善。

## 仍待补齐

A1/A2 需要在独立输出目录从头重跑并保留完整审计记录；本文件不把缺失记录推断为负结果。
