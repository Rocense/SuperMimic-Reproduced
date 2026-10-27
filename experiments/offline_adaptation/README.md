# 离线适配、教师和蒸馏检查

**2026-10-28 补充：** 下文描述审计时的旧实现。随后查阅官方 PMT 源码发现，其 batch 粗筛为 >200 mm 的根部/脚踝误差帧数超过 8 才拒绝；不是旧的单帧 5.5 cm 规则。四个 15 cm 探测样本按官方粗筛均通过。新增逐参考教师过滤、DAgger episode 控制及正式阶段依赖见 [更新后的流程](../pmt_protocol/README.md)；下文原测量值与历史结果保留。

本次按用户要求暂不训练。执行的是已有参考动作的独立几何审计、8 次 CPU TCRS 合成和参考姿态回放，没有启动新的 GPU 训练。

## 离线适配的实际效果

检查范围为夜间实验的全部 48 个 raw/adapted 配对：训练 40、验证 8，每段 256 帧、50 Hz。地形是原实验 2.5 cm 一级的旧 heightfield；训练中 26/40 窗口仅接触平地区域。以下统计不等同于大规模多地形评估。

| 指标 | raw 放到同一地形 | TCRS 后 |
|---|---:|---:|
| 全部帧的最大真实碰撞几何穿透 | 7.93 cm | 1.32 cm |
| 每帧最深真实穿透的均值 | 0.943 cm | 0.096 cm |
| 每段最大支撑脚 XY 滑动的均值 | 4.23 cm | 0.195 cm |
| 支撑期足底采样点离地超过 2 cm 的比例（片段均值） | 2.78% | 3.91% |
| 足端平均加速度 | 4.18 m/s² | 4.83 m/s² |

48/48 通过现有几何过滤；适配后上半身关节角与原始值完全一致，根部朝向保持不变。根部高度改变会使上半身的世界坐标发生位移，不能把“上半身关节不变”写成“上半身世界轨迹不变”。

适配有效减少了穿透和滑动，但不是无穿透或所有质量指标都改善。现有过滤检查脚尖/脚跟及小腿采样点；旧样本的采样点最大穿透仅约 0.080 cm，真实脚部 collision geom 的最大穿透却为 1.32 cm。这是稀疏点近似与真实碰撞形状之间的差异。接触、速度、力矩和动态可跟踪性也不能仅由离线几何通过判定。

另选出四个地形高度变化最大的独立源窗口（train/validation 各两个），分别在 2.5 cm 与 15 cm **实体台阶**上重新合成。每段仍保留全部 256 帧，合成参数相同，没有调低阈值。

| 样本 | 2.5 cm：最大足端目标误差 | 15 cm：最大足端目标误差 | 15 cm：真实最大穿透 |
|---|---:|---:|---:|
| train `walk1_subject1_f200_p1` | 2.41 cm | 14.39 cm | 1.32 cm |
| train `walk2_subject1_f1200_p1` | 2.99 cm | 21.12 cm | 5.26 cm |
| validation `walk1_subject5_f1200_p1` | 1.75 cm | 8.95 cm | 1.56 cm |
| validation `walk1_subject5_f200_p0` | 0.61 cm | 13.19 cm | 2.45 cm |

2.5 cm 四段全部通过，15 cm 四段全部被足端误差阈值拒绝；第二段同时超过采样点穿透阈值。这是明确挑选的跨台阶压力检查，不是全库通过率。它说明目前不能直接把台阶增高后沿用原来的“数据准备已完成”判断。最差样本的 15 cm 适配后支撑脚滑动达到 7.49 cm。

额外检查发现，部分高台阶上的脚尖/脚跟目标距离与刚性脚的固定距离不一致，造成最高约 3.28 cm 的不可消除点误差下界；这本身还不足以解释全部 8.95–21.12 cm 误差，不能把它当作已证明的唯一原因。

真实碰撞使用训练场景编译出的 MuJoCo 几何，在每一帧直接设置参考 qpos 后查询 robot–terrain 接触；不是动力学 rollout。四个高台阶最差碰撞帧另用完整 `mj_forward` 复核，结果完全一致。48 段的足端误差独立重算与原 manifest 最大差异为 1.71e-7 m。

本次离地/摆腿阈值是公开记录的审计口径：悬空为最低 toe/heel 采样点大于 2 cm；摆腿净空阈值 3.5 cm，包含起落脚邻近帧。不能直接把本次比例与论文不同样本、点集和阈值下的表格相比。

## Perceptive BFM 如何筛选，当前实现差在哪里

原文的 TCRS 使用多初值 IK 和连续性保护处理高误差/不连续解；随后先训练盲教师，再在蒸馏前剔除教师无法跟踪的参考。论文报告这一阶段剔除 7.4%（walk/run 约 5%，flip 约 20%），保留参考的教师完成率为 96.5%。见 [Algorithm 1、附录 A.6 Reference filtering](https://arxiv.org/html/2606.08059v3#A6)。原文没有把本仓库下面三个阈值规定为通用筛选标准。

本仓库当前实际做法：

1. 输入检查：50 Hz、G1 关节顺序、有限帧、地图范围、存在支撑接触。
2. 逐段几何过滤：最大足端目标误差 ≤5.5 cm，脚/小腿采样点最大穿透 ≤2.5 cm，腿部相邻帧关节跳变 ≤0.45 rad。三项全部满足才写入可训练集合。
3. 教师整体关口：train 完成率 ≥80% 且归一化回报 ≥0.65；validation 完成率 ≥60% 且回报 ≥0.60。

**第 3 步不是论文的逐参考可跟踪性过滤。** 目前没有把教师失败片段生成单独的排除清单：若整体门槛通过，编排器会把整个几何合格 train 目录交给 DAgger。`load_pairs()` 只读取几何 `accepted`，不读取 teacher 的逐片段评估。这里尚未实现与论文一致的第二轮动态过滤。

几何过滤还没有将真实 collision geom、悬空率、支撑滑动、速度/加速度和力矩可行性纳入验收。BONES 基础数据的名称/平地可行性过滤是另一回事，不能用其排除数代替 TCRS 或教师过滤效果。

相关代码：[`tcrs.py`](../../src/supermimic/perception/tcrs.py)、[`paired.py`](../../src/supermimic/perception/paired.py)、[`supervise.py`](../overnight_perceptive/supervise.py)。

## 当前教师训练和 policy 数量

每次教师实验是一个全身 29 关节 policy：一个 motion encoder 加一个 decoder，未按动作类别、上下半身或地形拆分专家。它读取 10 帧本体历史（930 维）和 10 帧 adapted 参考加 anchor（689 维），输出 29 维归一化关节位置动作。actor 不读取地图；世界模型与 critic 可以读取地图。

夜间实际训练的是三个独立候选模型，训练集和任务相同：

| 候选 | 起始 actor LR | critic | 作用 |
|---|---:|---|---|
| `teacher_lr1e5` | 1e-5 | 有 | 低学习率对照 |
| `teacher_lr3e5` | 3e-5 | 有 | 较高学习率对照 |
| `teacher_no_critic` | 3e-5 | 不使用 | 去掉 value bootstrap 的对照 |

三组都从随机初始化开始，使用相同 40 个 train 窗口和固定低台阶；不是三个技能专家，也不会在部署时切换三个控制器。选中一个合格教师后，原计划只蒸馏一个学生。本轮最好的模型来自第二组，但仍未达门槛。

每个候选中同时训练策略、世界模型和（启用时的）critic：仿真采集转移 → 用 replay 拟合世界模型 → 在世界模型内展开未来 16 步 → 通过 BPTT 的奖励梯度更新策略 → 更新 critic。critic 和世界模型都不是额外的控制 policy。

实际网络为 encoder/decoder `512,256,128`、latent 32、world model `512,512,512`，每组 512 个环境，策略 batch 256，WM batch 512×10、窗口 8；关闭观察噪声、重置噪声和域随机化，动作噪声 0.05。后期 actor LR 减半；最佳保存模型里的 actor 基准 LR 为 1.5e-5。这里采用 MLP+世界模型/BPTT；Perceptive BFM 使用 Transformer actor–critic + PPO。

当前完成率主要检查根部高度/倾斜，尚未包含论文所用的完整 anchor/end-effector tracking-loss 判定。95% train 完成率与 0.5535 回报、62.5% validation 完成率与 0.2208 回报应一起看，不能将“存活到片尾”解释为准确跟踪。

## 学生与教师的蒸馏

代码 [`distillation.py`](../../src/supermimic/perception/distillation.py) 使用 DAgger：

1. 冻结选中教师。学生复制教师的 motion encoder/decoder，增加高程图分支；两条地形残差门控初始化为零。零门控保证相同输入下原骨干输出不变，但 teacher 读 adapted、student 读 raw，所以两者初始输出不必相同。
2. 同一个仿真机器人状态上，teacher 看本体历史 + adapted 参考；student 看本体历史 + raw 参考 + 17×11 高程图及 mask。
3. 将 teacher 动作转换为同一组物理 PD 目标，再转换到 student 动作单位。将历史、raw 命令、地图和标签加入循环聚合缓冲区；MSE 只更新学生。
4. 每个环境、每一步随机选 teacher 或 student 控制。默认 teacher 概率在前 8,000 步从 1 降到 0，总蒸馏步数 10,000。student 控制时也在它实际访问到的状态上查询 teacher 标注。
5. 蒸馏后用 raw 命令、地图和 adapted 奖励目标继续做本仓库的世界模型/BPTT 微调。论文对应的最后阶段是 PPO。

动作坐标需要区别对待。Perceptive BFM 围绕参考关节角输出残差，因此标签为 `a_student* = q_tcrs + a_teacher - q_raw`（原文 Eq.6）。本仓库使用 `q_PD = offset + scale * action`，故实际标签是：

```text
a_student* = (offset_teacher + scale_teacher * a_teacher - offset_student) / scale_student
```

当前 teacher/student 的 default offset 和 scale 相同，换算退化成直接使用 teacher 动作；不能额外套一次 `q_tcrs - q_raw`，否则会重复改变物理目标。区别来自动作参数化，不是 raw/adapted 参考被混用。实现见 [`control.py`](../../src/supermimic/perception/control.py)。

本轮夜间教师没有过关，因此正式 DAgger、学生 BPTT、地图消融和感知模型导出都没有开始；已有的短流程检查不能替代这些阶段的训练结果。

## 复核文件

本地 `logs/offline_adaptation_20261027/` 下保留 `historical.json`、`probes.json`、`summary.json`、`probe_selection.json`、`feasibility_and_collision_crosscheck.json`、原始诊断日志、合成 NPZ、`height_probe.png` 及 `reference_comparison.gif`。回放为直接设置参考姿态，右侧故意展示一个被拒绝的 15 cm 样本，不代表控制器能力。

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python experiments/offline_adaptation/audit.py
PYTHONPATH=src:. MUJOCO_GL=egl \
  python experiments/offline_adaptation/visualize.py
```

下一步应先修正适配与碰撞几何的一致性、验证高台阶目标的可达性并加入逐参考教师过滤，再开始新的大规模感知训练。本次没有修改 TCRS 算法、验收阈值或历史数据来掩盖这些失败。
