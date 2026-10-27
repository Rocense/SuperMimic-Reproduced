# PMT 流程对齐，保留 SuperMimic MLP/BPTT

用户于 2026-10-28 明确选择保留 SuperMimic 的 MLP，只对齐数据筛选、教师/学生流程和蒸馏规则。当前仍按要求暂不训练。

## 来源与边界

固定对照 [Mondo-Robotics/PMT commit 1a92390077d329d22cbcdbf89c4a5c982003fbc8](https://github.com/Mondo-Robotics/PMT/tree/1a92390077d329d22cbcdbf89c4a5c982003fbc8) 和 [论文 v3](https://arxiv.org/html/2606.08059v3)。参考源码中的原始 PMT 部分采用 BSD-3-Clause，许可证见 [`third_party/pmt/LICENSE`](../../third_party/pmt/LICENSE)。不引入官方仓库中另有许可证的 BFM-Zero 实现。

| 项目 | 本分支决定 |
|---|---|
| 策略骨干 | 保留 SuperMimic MLP、命令接口和固定输入缩放 |
| 优化 | 保留世界模型、BPTT、critic 及 SuperMimic 可微奖励；不切换 PPO |
| 教师 | 一个全身、带 critic 的盲教师；actor 不读 height scan |
| 学生 | raw 命令 + 高程图/mask，零门控地形残差 |
| TCRS 粗筛 | 官方 batch 的根部/脚踝误差计数规则 |
| 教师后筛选 | 每条参考完整跟踪，剔除失败参考；记录 checkpoint/数据身份 |
| 蒸馏 | 冻结教师、物理目标对齐、MSE、按 episode 选择控制者、退火至学生 |
| 微调 | 从蒸馏学生及对应教师世界模型开始；骨干 LR 比例 0.3 |

这不是 Transformer/PPO 的逐项同构复现，也没有声称独立 TCRS 优化器与官方数值实现完全相同。迁移的是公开筛选规则与分阶段训练契约。官方公开库没有提供可独立运行的完整 teacher-pruning 脚本及论文所用的重复试验设置；本分支把论文“剔除教师无法跟踪的参考”落实为每条参考一次确定性全片段 rollout，并在筛选记录中明示。不能把未知设置声称为完全一致，也不能把论文 7.4% 当作固定删除比例。

## 两轮筛选

官方 TCRS `_record_solved_sequence` 计算每帧 `max(root_error, left_ankle_error, right_ankle_error)`；默认 `batch_reject_ik_mm=200`、`batch_reject_ik_bad_frames=8`，严格超过 200 mm 的帧数多于 8 才拒绝。对应文件为 `TCRS/stair_mppi/mppi_foot_planner_smooth.py` 的记录、batch 拒绝与参数定义部分。新生成数据默认采用该规则；旧 5.5 cm/2.5 cm/0.45 rad 判定留作 `legacy`，历史文件不改写。

教师后筛选采用官方 stepping-stone 任务的终止条件：anchor XYZ 距离 >0.35 m；anchor 坐标系下重力 Z 分量差 >0.8；经过 anchor/yaw 对齐后的脚踝或手腕位置误差 >0.35 m。任一条件在片段结束前触发即判失败。验证/测试集不参与删选；不存在整体 return 门槛、保留百分比配额或只评估成功子集。

筛选记录绑定完整候选清单、raw/adapted 文件哈希、配对 manifest 哈希、教师 checkpoint 哈希与阈值；缺少结果、重复结果、旧的仅存活评估、不同教师、不同配对集均不能通过正式 DAgger 检查。正式学生使用全部通过教师筛选的参考，并验证蒸馏 checkpoint 和世界模型来自同一教师。

**现有数据复核：** 原 15 cm 四个探测样本按旧阈值被拒绝；按官方根部/脚踝粗筛重判，四个都通过，均无 >200 mm 的帧。这不消除已测出的真实穿透，也不证明动态可跟踪。旧最佳教师 `teacher_lr3e5` checkpoint 在原 40 条训练参考上用新跟踪口径实测，仅 6/40 通过；26 条因 anchor 位置、8 条因末端位置失败。模型权重未更新，结果在 `logs/pmt_alignment_20261028/`。不能将这个只剩 6 条的结果包装成完成全量感知训练。

## 教师数量、参数与蒸馏

历史三个教师是学习率/critic 调试对照，不是 Perceptive BFM 必需的三位专家。新的主流程只使用一个带 critic 的教师。旧 `3e-5` 仅在小切片旧评估口径下占优，不是全量数据上的已证最优值。入口保留 SuperMimic 默认基准学习率 2e-4 和原尺寸 MLP；分布式学习率缩放沿用 SuperMimic 原实现。Perceptive BFM 教师 PPO 的 5e-4 不能视为 BPTT 的等价设置。

官方教师任务 `configs/task/walk_dance_bigmap.yaml` / `pmt_stepping_stone.yaml` 使用 `sensor: none`，其 Transformer actor 不读取 height scan。学生显式读取地图。本分支同样保持盲教师；地形进入适配参考以及 SuperMimic 的训练用世界模型/critic。

主蒸馏配置对照 `G1SteppingStoneVisionLatentAnchorDistillRunnerCfg`：每轮 24 个环境步，4 个学习 epoch，MSE、LR 1e-4、梯度裁剪 1.0；教师概率从 1 降到 0，100 个 iteration 对应 2,400 环境步；总 10,000 个 iteration 对应 240,000 环境步。每个 episode 固定控制者，仅 reset 时重选，使用学生确定性输出。正式 DAgger 的重置也采用相同的 PMT 跟踪失效判定，仍以 adapted 为监督目标；SuperMimic 的教师/学生 BPTT 训练终止模型保持原实现。环境数量与 minibatch 容量仍是本分支的资源配置，不冒充 48 A800 的实验设置。

动作目标转换保持物理 PD 目标相等。本仓库仍输出默认姿态周围的缩放动作，标签为 `(teacher_offset + teacher_scale * action - student_offset) / student_scale`，训练损失在物理弧度目标上计算。论文参考残差参数化下该公式等价于 `q_tcrs + a_teacher - q_raw`；不能同时保留固定 default offset 又额外加一次参考差。

## 正式顺序

1. 在 train/validation/test 源动作划分后分别生成 raw/adapted；新 manifest 记录 split，默认官方粗筛。
2. `scripts/train_perceptive.py` 训练一个带 critic 教师。
3. `scripts/filter_teacher_references.py` 生成该教师在 train 上的逐参考筛选记录。
4. `scripts/distill_perceptive.py --teacher-filter ...` 蒸馏，仅读取通过的参考。
5. `scripts/finetune_perceptive.py` 从该蒸馏模型和教师世界模型启动 BPTT。
6. 独立验证集用 `scripts/eval_benchmark.py --pmt-tracking-evaluation`，保留所有评估样本；随后地图消融与导出。

入口命令见根 README。`integration_test` 只允许很短、很小的流程测试，不能导出成正式部署产物；本轮没有运行任何教师训练、DAgger 或学生训练，只有冻结旧教师评估与 CPU 测试。
