# 2026-10-27 夜间训练实验

这是一次受时间预算约束的完整流程尝试：TCRS → 教师 → 质量门槛 → DAgger → 学生 BPTT → 地图消融与导出。启动不代表已经收敛；实际结果见运行目录的 `status.json` 和逐片段评估报告。

## 数据

使用本地已有的 LAFAN1 G1 50 Hz NPZ，未下载或提交原始数据。12 个 `walk*.npz` 各取从第 200、1200 帧开始的 256 帧窗口；将每个窗口起始骨盆的 XY 平移到 `(-0.75, 0)` 和 `(0.75, 0)` 两个位置。姿态、速度与时间轴保持不变，再运行 TCRS。

- 40 个训练窗口，8 个验证窗口，共 12,288 帧，约 246 秒；平移副本不算独立动作。
- subject 5 的两个原始行走序列及其所有窗口、副本只用于验证。这个验证集用于模型选择，不是额外的盲测集。
- 地图固定：12 m 高度场，0.1 m 分辨率，每级 0.025 m、共 4 级的往返台阶；同一原始动作在两个地图位置生成不同配对参考。
- 默认 TCRS 参数：96 个采样、12 轮摆腿优化、30 轮 IK。48/48 通过足端误差、地形穿透和相邻关节变化检查。
- `sources.json` 记录源文件 SHA-256、切片与平移；训练/验证 manifest 记录 raw/adapted SHA-256 和逐片段诊断。

本轮覆盖行走与单张地图，不代表完整 LAFAN1、未见地图或真机训练结果。

## 资源与训练计划

服务器使用四张 A6000。GPU 0–2 各跑一组独立教师，GPU 3 做评估；不是多 GPU 同步训练。三组共享同一数据、随机种子与网络规模，比较以下设置：

| 教师 | 策略学习率 | 世界模型先行更新 | Critic |
|---|---:|---:|---|
| `teacher_lr1e5` | 1e-5 | 800 | 开启 |
| `teacher_lr3e5` | 3e-5 | 500 | 开启 |
| `teacher_no_critic` | 3e-5 | 800 | 关闭 |

每组 512 个环境；世界模型隐藏层 `(512, 512, 512)`，策略编码器/解码器隐藏层 `(512, 256, 128)`。世界模型每次更新用 10 个 batch，每 batch 512，学习率 2e-4；策略 batch 256，BPTT horizon 16。启用 anchor observation，关闭观测/重置噪声和动力学随机化，动作探索噪声 0.05。

确定性评估从每个片段首帧开始。教师须同时满足训练完成率 ≥80%、平均归一化回报 ≥0.65，验证完成率 ≥60%、回报 ≥0.60，才启动 DAgger。回报按实际存活步数归一化，因此必须结合完整片段完成率与完成帧比例阅读。

DAgger 运行 10,000 个向量化环境步，512 个环境，每步做两次监督更新，教师控制概率在第 8,000 步降到零。学生随后加载蒸馏策略和同地图教师世界模型，以 1e-5 学习率进行 BPTT 微调。最终评估正常地图、全零地图和打乱地图，并导出三输入 TorchScript 控制策略。

`plan.json` 固定了本次实验的北京时间：教师首轮至 06:15，学生训练或未达标教师续训至 09:15，总队列硬截止 09:50。若教师第一轮没有达标，使用有效 checkpoint、将学习率减半继续教师训练。未达门槛不会自动标记学生训练完成。重新运行时必须设置新的输出目录和未来的截止时间。

## 启动与检查

从仓库根目录运行；先安装主 README 指定依赖，并保证 `.venv/bin/python` 可用。数据生成与监督器都使用本实验固定目录 `data/overnight_walk_20261027`、`logs/overnight_20261027`。

```bash
PYTHONPATH=src .venv/bin/python experiments/overnight_perceptive/prepare_data.py /path/to/lafan1_npz
PYTHONPATH=src .venv/bin/python experiments/overnight_perceptive/prepare_data.py
mkdir -p logs/overnight_20261027
cp experiments/overnight_perceptive/plan.json logs/overnight_20261027/plan.json
.venv/bin/python experiments/overnight_perceptive/supervise.py
```

部署时通过独立进程会话启动监督器，将输出重定向到 `supervisor.log`；服务器训练不依赖 SSH 连接保持打开。查看：

- `status.json`：阶段、子进程、GPU 分配、评估记录和最佳模型。
- 各教师/学生的 `console.log`、`progress.json`、TensorBoard：训练更新、时间和模型误差。
- `evaluation_*/train.json`、`validation.json`：逐片段真实完成步数和失败状态。
- `selected_teacher.pt`、`best_teacher.pt`、`best_student.pt`：选择结果；只有实际产生后才存在。
- `final_evaluation/`、`student.ts`、`student.json`：最终评估和控制接口。

保存使用临时文件再原子替换。每组保留初始、最新和最近两个编号 checkpoint；另保留选中的教师及最佳模型。仅移除本实验的旧编号快照和已评估临时副本，不删除源码、数据或其他项目结果。磁盘不足时停止，并保留有效快照。

`completed` 表示所有计划阶段执行完毕，能力是否达标仍需阅读评估结果；`teacher_quality_not_met` 表示教师未达门槛、学生阶段未执行。`failed` 必须检查具体子进程与日志后再恢复，监督器不会盲目重复启动。

## 运行中修正

02:56 的检查发现，按最新 checkpoint 的修改时间选择评估对象，会偏向更新较慢的教师；更新快的无 critic 教师可能长时间得不到评估。现改为优先选择最久未评估的教师，并把评估所用的源 checkpoint 路径写入记录。

监督器增加 `--resume-teachers`，仅用于教师阶段的进程接管：按保存状态核对 worker 的用户、命令、job 路径、进程组和进程启动标识，保留现有学习器及其训练状态，恢复评估计数和最佳模型。它不用于启动重复训练，也不支持从学生阶段盲目恢复。2026-10-27 约 03:01 完成交接，三组教师 PID 保持不变。

回归检查：`python -m pytest -q experiments/overnight_perceptive/test_supervise.py`，3 项通过，覆盖评估公平性、在运行的 worker 接管与完成状态读取、拒绝无关 PID。
