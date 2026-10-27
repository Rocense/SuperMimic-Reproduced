# Perceptive SuperMimic

**TCRS → 盲教师 → DAgger 感知学生 → 世界模型/BPTT 微调。**

本分支把地形感知训练接入 SuperMimic：人类动作经离线 TCRS 转换为地形适配参考，教师学习跟踪适配动作，学生学习从原始动作和机器人侧高程图恢复这些控制行为。部署时只运行学生策略。

**训练状态（2026-10-27）：** 夜间实验已结束。最佳教师的训练/验证片段完成率为 95% / 62.5%，但平均归一化跟踪回报为 0.5535 / 0.2208，未通过质量门槛。因此本轮没有启动 DAgger、感知学生微调或正式导出；目前不能据此宣称已训练出可用感知学生。完整结果见 [验证记录](VALIDATION.md)。

这是基于 [SuperMimic-Reproduced](https://github.com/Rocense/SuperMimic-Reproduced) 的研究实现。采用 [Perceptive BFM](https://arxiv.org/abs/2606.08059) 的分阶段训练路线，保留 SuperMimic 的 MLP 与可微世界模型/BPTT；TCRS 是本分支的独立工程实现。参考生成参数、网络和优化方法与原论文并不完全相同，运动能力需要收敛训练和独立评估。

## 训练与部署结构

```text
平地 G1 动作 + 地形
        │
        ▼
离线 TCRS：接触检测 / 落脚锁定 / 摆腿优化 / 根部重建 / 多点 IK
        │
        ├── raw/*.npz       原始参考
        └── adapted/*.npz   地形适配参考，帧数与时间轴一致
                    │
                    ▼
盲教师：本体历史 + adapted 参考 → 关节目标
                    │ 在学生访问到的状态上查询教师
                    ▼
DAgger：本体历史 + raw 参考 + 高程图 → 感知学生
                    │
                    ▼
地形条件世界模型 + BPTT 微调
                    │
                    ▼
部署：本体历史 + raw 参考 + 高程图 → 学生 → 关节 PD
```

| 阶段 | 策略命令 | Actor 地形输入 | 训练监督 |
|---|---|---|---|
| 教师 | TCRS 适配动作 | 无 | 适配参考、地形接触与跟踪奖励 |
| DAgger | 原始动作 | 有 | 同一机器人状态下的教师关节目标 |
| 学生微调 | 原始动作 | 有 | 适配参考奖励和地形条件世界模型 |
| 部署 | 原始动作 | 有 | 无教师、无在线 TCRS |

教师虽然是盲策略，其训练用世界模型和 critic 仍读取地形。学生的原始命令、教师的适配命令、训练奖励目标分别传递；适配参考不会混入学生的命令输入。

## 已实现的组件

- **TCRS**：带迟滞的接触检测、支撑脚锁定、考虑脚尖/脚跟的采样式三次样条摆腿优化、支撑加权根部高度重建、12 个腿部关节的多点阻尼 Jacobian IK、脚/小腿穿透修复、多初值重试和连续性约束。保留原始根部朝向与上半身关节；不重排动作时间轴。
- **参考质量过滤**：记录足端误差、穿透和关节跳变。未通过阈值的片段放入 `rejected/`，不会自动成为训练数据。NPZ 对及 manifest 使用 SHA-256 检查帧对齐和文件一致性。
- **地形感知**：17×11 高程网格，187 个相对高度与 187 个有效性标记，共 374 维。网格随根部位置和 yaw 更新，支持高度噪声与丢点。普通高度场共用高度样本及三角剖分；新楼梯使用实体 box 和相同的解析踏面边界，保留垂直立面。
- **感知策略**：保留原跟踪骨干，加入地形编码器、潜变量残差及动作残差，门控初始化为零。
- **地形条件世界模型**：每个预测状态重新查询局部地图；不把 rollout 起点的裁剪图冻结到整个预测过程。
- **DAgger**：缓存、聚合教师标签，在学生访问到的状态上继续标注；教师控制概率逐步降到零。动作标签通过实际 PD 位置目标换算。
- **微调与推理**：支持加载蒸馏策略与教师世界模型、分阶段续训、地图消融、回放及独立 TorchScript actor 导出。

## 环境

沿用上游 `pyproject.toml` 与 `uv.lock`：Python 3.12、PyTorch、MuJoCo/MuJoCo Warp，以及固定版本的 mjlab。需要 NVIDIA GPU 执行训练；TCRS 和数值测试可以在 CPU 上运行。

```bash
git clone --branch perceptive-supermimic https://github.com/Rocense/SuperMimic-Reproduced.git
cd SuperMimic-Reproduced
uv sync --group dev
export PYTHONPATH="$PWD/src"
```

下面用 `python` 表示已激活环境中的解释器；使用 uv 时可换成 `uv run python`。应选择与机器驱动兼容的 PyTorch/CUDA 组合。不要把另一份 SuperMimic checkout 留在 `PYTHONPATH` 中。

## 数据集与准备流程

这套流程需要 **G1 动作数据和程序化地形**。起点是平地动作，经 TCRS 生成地形适配参考；不需要提前采集“每个动作在每种地形上”的真人示范。当前感知来自仿真高度场，不依赖 RGB/深度视频数据集。

| 数据源 | 本分支用途 | 输入形式 |
|---|---|---|
| [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) | 接通流程、小规模训练与消融；上游使用 40 个 G1 片段 | `g1/` 下 30 Hz 的 G1 CSV |
| [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) | 扩展动作覆盖；数据源提供 142,220 个动作，含镜像版本 | G1 包中的 120 Hz CSV |
| 自有 G1 动作 | 追加动作或遥操作参考 | 满足下述规范的 50 Hz NPZ，或可转换的 G1 CSV |

数据源规模不等于本分支已验证的规模。四阶段集成检查使用 `walk1_subject1` 的第 200～327 帧，共 128 帧。2026-10-27 的服务器实验另准备了 40 个训练和 8 个 subject 5 验证行走窗口，均通过 TCRS 几何检查；训练计划和质量门槛见 [夜间实验说明](experiments/overnight_perceptive/README.md)。尚未完成完整 LAFAN1 或 BONES-SEED 的感知训练。

### LAFAN1：下载 G1 动作并转成 50 Hz NPZ

环境已声明 `huggingface-hub` 依赖。以下命令只选择 G1 CSV 和数据说明：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lvhaidong/LAFAN1_Retargeting_Dataset",
    repo_type="dataset",
    allow_patterns=["g1/*.csv", "README.md", "LICENSE"],
    local_dir="data/raw/lafan1",
)
PY

python scripts/data/csv_to_npz_batch.py \
  --input-dir data/raw/lafan1/g1 \
  --output-dir data/npz/lafan1 \
  --input-fps 30 --output-fps 50 --workers 4
```

LAFAN1 G1 CSV 没有表头，每帧为根部位置、根部四元数 **xyzw** 和 29 个关节角。本仓库转换器按米/弧度读取，并输出四元数 **wxyz** 的 NPZ；会重采样、重算速度并运行 G1 正运动学。不要使用 H1/H1_2 文件替代 G1 文件。源格式和关节顺序见 [数据卡](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset#dataset-collection-pipeline)。

转换后检查终端中的失败列表和 `data/npz/lafan1/manifest.csv`，实际片段数和时长以该清单为准。默认不会覆盖已有 NPZ。

### BONES-SEED：访问、转换与过滤

先在 [数据页](https://huggingface.co/datasets/bones-studio/seed) 完成访问申请并在本机登录 Hugging Face。下载代码使用本机已有凭据，不要把 token 写入代码或提交到仓库。

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="bones-studio/seed",
    repo_type="dataset",
    filename="g1.tar.gz",
    local_dir="data/raw/bones_seed",
)
PY

tar -xzf data/raw/bones_seed/g1.tar.gz -C data/raw/bones_seed

python scripts/data/csv_to_npz_batch.py \
  --input-dir data/raw/bones_seed/g1/csv \
  --output-dir data/npz/bones_seed \
  --input-fps 120 --output-fps 50 --workers 4

python scripts/data/filter_bones_seed.py \
  --npz-dir data/npz/bones_seed --workers 4
```

转换器识别 BONES 带表头的 G1 格式，并处理厘米、角度和欧拉角到本仓库单位/四元数的转换。这里读取的是 G1 CSV，SOMA BVH 不能直接作为 G1 控制参考。发布包目录结构见 [官方说明](https://huggingface.co/datasets/bones-studio/seed#dataset-structure)。

过滤脚本输出 `manifest_filtered.csv`，列为 `name,frames,keep,reason`，检查名称和几何可行性。**它只写清单，不移除 NPZ；TCRS 也不会自动应用这份 CSV。** 可以建立仅包含 `keep=1` 的子集，再交给 TCRS：

```bash
python - <<'PY'
import csv
from pathlib import Path
source = Path("data/npz/bones_seed")
subset = Path("data/npz/bones_seed_kept")
subset.mkdir(parents=True, exist_ok=True)
with (source / "manifest_filtered.csv").open() as stream:
    for row in csv.DictReader(stream):
        if row["keep"] != "1":
            continue
        path = source / (row["name"] + ".npz")
        link = subset / path.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(path.resolve())
PY
```

把后续 `synthesize_terrain.py --npz-dir` 指向该子集。该过滤器来自平地复现流程，仍应结合目标动作和本分支的 TCRS 质量诊断选片；上游报告的过滤数量不是本分支的实测结果。

### 目录与数据契约

```text
data/
  raw/lafan1/g1/*.csv             下载的源动作
  npz/lafan1/*.npz                重采样后的 G1 动作
  npz/lafan1/manifest.csv          转换统计
  tcrs_train/
    raw/*.npz                    学生的命令来源
    adapted/*.npz                教师命令和训练奖励目标
    rejected/*.npz               未通过 TCRS 检查的诊断产物
    manifest.json                地形、合成参数、质量指标、文件哈希
  tcrs_validation/                独立验证片段/地形生成的数据集
```

标准 NPZ 字段如下，`T` 为帧数，`B` 为文件中保存的刚体数：

| 字段 | 形状 | 约定 |
|---|---|---|
| `fps` | `(1,)` | 50 Hz |
| `joint_pos` / `joint_vel` | `(T,29)` | 弧度 / 弧度每秒 |
| `body_pos_w` | `(T,B,3)` | 世界坐标，米 |
| `body_quat_w` | `(T,B,4)` | 单位四元数，wxyz |
| `body_lin_vel_w` / `body_ang_vel_w` | `(T,B,3)` | 世界坐标，m/s / rad/s |
| `joint_names` | `(29,)` | 必须匹配 `data/motion_convert.py` 的 `G1_JOINT_NAMES` |
| `body_names` | `(B,)` | 包含 pelvis 及训练使用的全部跟踪刚体 |

TCRS 额外保存接触标记、足端目标和根部偏移。`raw/` 与 `adapted/` 必须同名、同帧数、同关节顺序和同时间轴，不能单独裁剪或重排其中一侧。训练入口根据配对 manifest 校验哈希；修改已生成的文件会明确报错。

**当前 teacher/student 阶段只支持上述 NPZ 配对数据集。** 上游 `build_library.py` 生成的紧凑 `.pt` library 保留给平地训练；当前配对阶段拒绝 `--data.library` 和 `--data.manifest`，统一使用 `--data.paired-dir`。

### 训练/验证划分与地形泛化

建议先按源片段或采集组划分动作，再各自生成 TCRS；不要把同一长动作的相邻窗口随机分到训练与测试两边。LAFAN1 可按人物或完整片段留出；BONES 应将同源镜像动作及同一采集组放在同一侧，可结合其 `take_name`、`take_actor`、`is_mirror` 元数据分组。该分组策略需要自行保存清单，本仓库目前没有自动划分脚本。

分别记录以下两种评估，避免混成一个泛化数字：

- **未见动作**：留出的源片段，在指定地形上生成独立配对数据。
- **未见地形**：使用新的 terrain seed/几何配置重新生成配对数据，保持 actor 的高程图输入尺寸一致。

例如用验证动作目录和 seed 91 生成验证集：

```bash
python scripts/synthesize_terrain.py \
  --npz-dir data/npz/lafan1_validation \
  --output data/tcrs_validation \
  --terrain.kind boxes --terrain.height 0.08 --terrain.seed 91
```

验证目录需要先根据保存的划分清单建立。评估时传入 `--paired-dir data/tcrs_validation`。当前每次训练使用一个固定地形，因此多个数据集目录不能直接混成多地图训练集；多地图 replay 还需扩展地图 ID 和坐标变换。

实验记录应包含数据源版本、源片段清单、原始/输出 FPS、过滤配置、TCRS 参数和地形种子。源数据与生成数据遵循各自数据许可，不由本仓库的 MIT 代码许可证覆盖；许可与访问条件见对应数据页。`data/`、`logs/`、NPZ 和 checkpoint 均不提交到 Git。

## 1. 离线生成成对数据

```bash
python scripts/synthesize_terrain.py \
  --npz-dir data/npz/lafan1 \
  --output data/tcrs_boxes \
  --clips walk1_subject1 walk1_subject2 \
  --terrain.kind boxes --terrain.height 0.08 --terrain.seed 17
```

输出目录：

```text
data/tcrs_boxes/
  manifest.json
  raw/        # 被接受片段的原始参考
  adapted/    # 同名、同帧数的 TCRS 参考
  rejected/   # 未通过几何质量检查的诊断产物
```

`--start-frame` 和 `--max-frames` 可用于短片段检查，两侧会使用完全相同的裁剪区间。输出目录必须为空，避免覆盖已有数据。地形可选 `flat / boxes / stairs / slope`；`height` 对 boxes 是最大高度、对 stairs 是单级高度、对 slope 是斜率（m/m）。

新楼梯默认使用 `--terrain.stairs-surface boxes`。例如单级 15 cm、踏面 50 cm、4 级最高 60 cm：

```bash
python scripts/synthesize_terrain.py \
  --npz-dir data/npz/lafan1 --output data/tcrs_stairs_15cm \
  --clips walk1_subject1 walk1_subject2 \
  --terrain.kind stairs --terrain.stairs-surface boxes \
  --terrain.height 0.15 --terrain.stair-width 0.5 --terrain.stair-count 4
```

生成后还须检查轨迹确实经过台阶。当前楼梯布局沿世界 X 轴在两侧先升后降，中央留平地；仅有行走动作不保证发生上楼。更换高度或表面类型必须生成新的 TCRS 配对数据，再训练/评估。

TCRS 产生运动学参考。几何检查通过后，仍需用训练好的教师验证动态可跟踪性；教师持续失败的片段应从后续 `--data.clips` 选择中排除。

## 2. 训练盲教师

```bash
python scripts/train_perceptive.py \
  --stage teacher --run-name terrain_teacher \
  --data.paired-dir data/tcrs_boxes \
  --schedule.max-hours 1
```

教师读取 `adapted/`，actor 不使用高程图，世界模型和 critic 使用地形。训练循环先拟合世界模型，再开启 BPTT。`train_perceptive.py` 默认使用 256 个环境和 batch size 256；这些是起始配置，不是经收敛验证的最佳参数。

可以用 `--init-policy /path/to/flat_checkpoint.pt` 加载兼容的平地策略。网络尺寸、goal 配置和关节接口必须匹配。此分支保留目标仓库原有的固定输入缩放和训练迭代口径，不能直接加载其他分支任意架构的 checkpoint。

训练后检查教师对配对参考的完成情况：

```bash
python scripts/eval_benchmark.py \
  --ckpt logs/terrain_teacher/ckpt_latest.pt \
  --paired-dir data/tcrs_boxes \
  --out logs/terrain_teacher/evaluation.json
```

## 3. DAgger 蒸馏感知学生

```bash
python scripts/distill_perceptive.py \
  --teacher logs/terrain_teacher/ckpt_latest.pt \
  --paired-dir data/tcrs_boxes \
  --output logs/terrain_student/distilled.pt \
  --steps 10000 --decay-steps 8000
```

学生继承教师的跟踪骨干，地形残差从零门控开始。环境给学生原始参考；教师在相同状态、相同动作帧上接收适配参考。标签缓冲区聚合历史数据，执行控制逐步由教师转交给学生。

## 4. 世界模型/BPTT 微调学生

```bash
python scripts/train_perceptive.py \
  --stage student --run-name terrain_student_finetune \
  --data.paired-dir data/tcrs_boxes \
  --init-policy logs/terrain_student/distilled.pt \
  --init-world-model logs/terrain_teacher/ckpt_latest.pt \
  --schedule.max-hours 1
```

学生命令保持原始参考，奖励/终止判断使用离线适配参考。教师的世界模型可以作为初始化；地形必须与配对数据一致。`--resume` 用于同阶段、同地形续训，不能与初始化参数混用。

`max-hours` 是示例训练预算，不表示在该时间内能够达到可靠运动效果。

## 回放、消融与未见地形

先检查楼梯形状，不加载策略：

```bash
python scripts/view_terrain.py --height 0.15 --stair-width 0.5 --stair-count 4
```

这是静态几何预览，机器人仅作为尺寸参照。添加 `--output logs/stairs_15cm.png` 可以保存图片。

旧 checkpoint/manifest 没有 `stairs_surface` 字段时，仍使用原来的 `heightfield`，以复现原实验。原夜间楼梯每级 2.5 cm、网格间距 10 cm，立面被三角面插值成小斜坡；这些结果不代表垂直楼梯上的训练成绩。`play.py` 可显式添加 `--stairs-surface boxes --terrain-height 0.15` 查看新几何下的行为，但这属于诊断覆盖，旧策略和旧参考并未按新地形训练，不能当作正式评测。

仅运行学生和原始参考即可回放：

```bash
python scripts/play.py \
  --ckpt logs/terrain_student_finetune/ckpt_latest.pt \
  --npz-dir data/tcrs_boxes/raw --num-envs 1
```

量化评估使用配对数据作为训练外的目标依据：

```bash
python scripts/eval_benchmark.py \
  --ckpt logs/terrain_student_finetune/ckpt_latest.pt \
  --paired-dir data/tcrs_boxes --map-ablation none \
  --out logs/terrain_student_finetune/eval_map.json

python scripts/eval_benchmark.py \
  --ckpt logs/terrain_student_finetune/ckpt_latest.pt \
  --paired-dir data/tcrs_boxes --map-ablation zero \
  --out logs/terrain_student_finetune/eval_zero_map.json
```

`shuffle` 消融要求至少两个片段和两个并行环境。未见地形评估应使用不同种子重新生成一个配对数据集，再通过 `--paired-dir` 指定。不能把旧地形的适配参考用于新地形奖励。

评测保留上游回报统计，同时必须关注 `failed`、实际完成帧数及物理接触情况。不同地形下的奖励目标不同，回报不能替代跨地形成功率。

## 下游 controller 接口

这个模型本身是全身策略，下游接关节位置/PD 控制器。默认以 50 Hz 输出 29 维归一化动作：

```text
a = actor(proprio_history, raw_motion_goal, terrain_scan)
q_des = default_joint_position + action_scale * a
tau = Kp * (q_des - q) - Kd * dq
```

`proprio_history` 默认 930 维，按观测项排列 10 帧历史；`raw_motion_goal` 默认 680 维，启用 anchor 观测后为 689 维；`terrain_scan` 为 374 维。具体尺寸以 checkpoint 和导出的 JSON 为准。

蒸馏时先求教师的实际关节目标，再转换为学生动作：

```text
q_des_teacher = offset_teacher + scale_teacher * action_teacher
label_student = (q_des_teacher - offset_student) / scale_student
```

当前教师和学生使用相同的默认姿态与缩放，因此标签等价于直接对齐动作。这里没有把动作解释成“原始参考关节角上的残差”。

导出：

```bash
python scripts/export_perceptive.py \
  --ckpt logs/terrain_student_finetune/ckpt_latest.pt \
  --out logs/terrain_student_finetune/actor_torchscript.pt
```

同名 JSON 包含输入维度、地图约定、关节顺序、动作缩放、默认位置和控制周期。部署端负责状态历史、参考动作窗口、地图坐标变换、机器人 SDK 通信及 PD 增益；导出模型不包含建图或硬件驱动。

## 验证

```bash
python -m pytest tests/test_perceptive.py tests/test_perceptive_pipeline.py -q

python scripts/smoke_pipeline.py \
  --npz-dir data/npz/lafan1 --output logs/pipeline_smoke
```

`smoke_pipeline.py` 使用一段行走数据依次执行 TCRS、短教师训练、DAgger、短学生微调和 checkpoint 重载；添加 `--compile` 可检查编译路径。输出目录必须是新的目录。验证记录见 [VALIDATION.md](VALIDATION.md)。短训练产物用于接口检查，不能当作训练好的机器人控制模型。

## 当前范围

- 每个数据集/训练运行使用一个固定地形，所有独立仿真 world 共享同样的局部坐标定义。多地图联合训练尚需在 replay 中保存地图 ID 和坐标变换。
- 目前输入是带噪声和丢点的理想高程图。真实深度/LiDAR 建图、遮挡、时延和真机状态估计需要独立集成。
- TCRS 当前面向有足部支撑的运动，使用脚尖、脚跟和小腿采样点检查地形；没有覆盖完整接触丰富动作、悬空多层表面或动态障碍。
- 尚未提供收敛后的地形泛化结果、长时训练吞吐或真机结果。

## 代码位置与来源

- `src/supermimic/perception/tcrs.py`：离线参考合成。
- `perception/paired.py`：数据对齐、完整性校验和阶段配置。
- `perception/env.py`：原始命令与适配监督的环境接口。
- `perception/distillation.py`、`perception/control.py`：DAgger 和 PD 目标对齐。
- `perception/terrain.py`：碰撞高度场与可微高程查询。
- `policy/`、`world_model/`、`imagination/`：感知策略、地形条件动力学与 BPTT。

本分支基于 Rocense/SuperMimic-Reproduced 的 `d9be385`，保留上游 [MIT LICENSE](LICENSE)。研究背景：[SuperMimic](https://arxiv.org/abs/2610.28480)、[Perceptive BFM](https://arxiv.org/abs/2606.08059)。本分支的验证结果与上游论文报告分别记录。
