# 完整动作语料接入（2026-10-27）

夜间实验只使用约 122.88 秒独立行走动作，没有落实完整动作语料训练。它的结果不能代表全量 SuperMimic 或 Perceptive BFM。目前已核实服务器有完整 BONES 数据副本；本轮将其接入独立的全量实验目录。

本目录接入**完整过滤后的 BONES-SEED**，同时保留全部 40 个 LAFAN1 G1 长序列。准备脚本逐一核对动作名称、帧数、连续索引、所有参考状态的有限性、四元数及 SHA-256。数据使用原文件链接，避免占满磁盘；不会删改原文件，也不做固定长度切片。

## 数据范围和状态

| 数据 | 实际覆盖 | 状态 |
|---|---|---|
| BONES 原始发布包 | 142,220 段（包含镜像） | 既有服务器原件保留 |
| BONES 可用于平地的过滤结果 | 129,943 段；47,475,680 帧；263.7538 小时；50 Hz | 完整紧凑库，约 6.75 GB |
| 过滤排除 | 12,277 段，其中名称过滤 12,083、几何过滤 194 | 原始数据保留，排除原因在清单中 |
| LAFAN1 | 40 个完整 G1 长序列 | 全部接入，包含行走、跑步、舞蹈、跳跃、格斗、倒地起身 |
| 全量 TCRS / 教师 / DAgger / 感知学生 | 尚未完成 | 不把原始库就绪计为感知训练完成 |

BONES 镜像版本计作不同发布片段，263.75 小时也是发布片段口径。这里的过滤规则是对 SuperMimic 附录 E 的近似实现，**129,943 不是论文的 129,514**，不能把数字写成一致。

主基础实验用完整过滤后的 BONES 训练，全部 40 个 LAFAN1 片段用于跨数据源评估；不把 LAFAN1 评估片段加入这次 BONES 训练。另提供使用全部 40 个 LAFAN1 的独立训练配置，仅供基础跟踪对照；该配置的 LAFAN1 结果是训练集结果，不能叫 held-out validation。

## 服务器接入

独立项目目录：`/data/zclin/RL/perceptive_supermimic_full_corpus_20261027`。
只读数据来源：`/data/zclin/RL/Supermimic_mod/data`。
旧夜间结果和恢复目录继续保留。

```bash
cd /data/zclin/RL/perceptive_supermimic_full_corpus_20261027
export PYTHONPATH="$PWD/src"

# 完整 CPU 校验、建立文件链接并生成无切片的配置。
.venv/bin/python experiments/full_corpus/prepare.py \
  --source /data/zclin/RL/Supermimic_mod/data \
  --output data/full_corpus

# 默认只检查输入，不启动训练。
.venv/bin/python experiments/full_corpus/train.py \
  --config data/full_corpus/bones_flat_config.json

# 在已分配的 GPU 上启动；例中 0 需要替换为实际分配的卡号。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/full_corpus/train.py \
  --config data/full_corpus/bones_flat_config.json --execute

# 训练后：完整训练库评估，以及完整 LAFAN1 跨源评估。
.venv/bin/python scripts/eval_benchmark.py \
  --ckpt logs/bones_full_flat/ckpt_latest.pt \
  --library data/full_corpus/bones_seed_compact.pt
.venv/bin/python scripts/eval_benchmark.py \
  --ckpt logs/bones_full_flat/ckpt_latest.pt \
  --npz-dir data/full_corpus/lafan1
```

`data/full_corpus/inventory.json` 记录实际数据、哈希及未完成阶段。`bones_flat_config.json` / `lafan1_flat_config.json` 为完整训练参数。预检不是训练成功或收敛证据。

基础配置恢复原尺寸网络和默认优化设置：编码器/解码器 `2048,1024,512,256`，世界模型 `2048,2048,2048`，latent 32；4,096 个环境，世界模型 batch 2,048 × 10、窗口 8，策略 batch 4,096、BPTT horizon 32，学习率 2e-4，320,000 次环境步，UTD 0.4；开启观察噪声、重置扰动和域随机化。不沿用夜间实验缩小的网络和短窗口。checkpoint 存档间隔改为 20,000 步以控制磁盘增长，不改变训练步数。

用户曾指定使用四卡，随后明确要求暂不训练、先检查离线适配与蒸馏机制。当前没有启动新的 GPU 训练。后续启动前还需按实际 GPU 显存做运行容量验证；这些是待运行配置，不是已经完成的训练设置验证。

## Transformer 与复现边界

[SuperMimic 原文附录 C、表 A.4](https://arxiv.org/pdf/2610.28480#page=15) 的 dynamics、motion encoder、shared decoder、critic 均为 MLP，没有 Transformer。采用历史窗口不等于使用 Transformer。

[Perceptive BFM §3.3、附录 A.7](https://arxiv.org/html/2606.08059v3#A7) 的教师和学生采用 Transformer 跟踪骨干；感知分支用地图 CNN 和 query-conditioned map Transformer，经零初始化门控作用于意图特征和动作。其训练路线为 TCRS → PPO 教师 → DAgger → PPO 微调。

该文消融了 TCRS、地图、门控、动作目标坐标对齐等组件，没有给出保持其他条件一致的纯 MLP 对照以证明 Transformer 是必要条件。本分支继续采用 SuperMimic 的 MLP 与世界模型/BPTT；这是迁移感知训练方法，不能称为 Perceptive BFM 的逐项同构复现。不能从当前切片实验的低 validation 回报推断“缺少 Transformer 导致失败”。

## 全量感知训练尚需补齐的工作

当前基础跟踪器能加载紧凑 `.pt`；现有 TCRS、教师和 DAgger 的配对读取器仍要求逐文件 NPZ。单一固定地形也不足以覆盖 263.75 小时多样动作。需要把完整原始库接入分片、可恢复的 TCRS 数据生成和配对读取，按源动作划分评估，再按动作/地形适配性记录接受或拒绝原因。不能直接把完整压缩库路径当作已经生成的 TCRS 配对库。

平地过滤中依赖楼梯、椅子等外部支撑的原始动作保留在源包中。它们应在对应支撑几何和合适的接触建模下另行纳入，不能放到平地无支撑任务中硬训，也不能把当前仅处理腿部接触的 TCRS 当作已支持手部接触重定向。
