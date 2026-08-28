# Mandarin Reverberation-Robust ASR

本项目研究普通话远场混响条件下的 ASR 鲁棒性，核心问题是：

> 可控混响如何影响 CER；传统单/多通道 WPE 与 Whisper 多条件 LoRA 适配是互补、
> 冗余，还是会引入前端—模型失配？

项目使用 AISHELL-1 干净普通话、Pyroomacoustics 四通道 RIR、NARA-WPE 和固定 revision
的 `openai/whisper-small`。RT60 是主实验变量，DRR 用于解释同一 RT60 档内的识别差异，
声源—阵列距离只作为几何协变量。

## 当前状态

- 已完成 AISHELL-1 下载、完整性审计和可复现数据划分。
- 已冻结约 20 小时训练集与 dev/test 评测子集。
- 已实现四通道 UCA 几何、可控 RT60 RIR 生成、DRR 估计与多通道卷积；已有 200 条 dev RIR。
- 已生成并联合校验正式 train/dev/test RIR：分别为 1,000/200/300 组、房间完全隔离；test 包含
  60 个同几何 family，每个 family 配对五档 RT60。
- 已实现 Raw、单通道 WPE（10/40 taps）和多通道 WPE（10 taps）四条前端分支。
- 已实现普通话文本归一化、CER、替换/删除/插入统计和 paired bootstrap。
- 已验证冻结 Whisper GPU 推理和 LoRA 前向/反向链路。
- 已完成协议校验、数据泄漏检查和自动测试。
- 已完成 500 条开发集 Clean+Raw 五档 RT60 基线；Raw CER 从 13.56% 单调上升至 27.36%。
- 已实现 Clean/MCT 训练 Dataset、epoch 级确定性 RIR 采样、Whisper batch collator、可恢复
  优化循环和带 clean CER 安全门的 checkpoint 选择。
- 已完成正式 WPE 开发集消融；M-WPE-10 在五档 RT60 均取得最低 CER，且在
  `RT60>=0.4 s` 显著优于两种单通道控制。
- 已完成 500 条无混响/仅直达声输入审计：响度和传播链路未造成显著 CER 偏差，但无混响时
  M-WPE-10 没有收益且存在插入错误风险，因此只作为已知混响条件的实验前端。
- 已完成 greedy/beam 解码对照：Beam-5 改善 clean 子集绝对 CER，但不改变 M-WPE 的收益方向；
  为保持全矩阵可比性，主实验继续统一 greedy，Beam-5 只作最佳模型的可选二级结果。
- 已完成 500 条×5档×4前端的 direct-target SI-SDR/STOI 审计；M-WPE 的信号与 CER 收益总体
  同向，但单通道 WPE 存在“信号指标改善、CER 恶化”，证明前端必须按下游指标选择。
- 已生成严格嵌套的 5/10/20 小时训练 manifest，并完成正式 dev evaluator、可恢复训练 CLI 与
  clean CER 安全门；W0 `dev_model` 冻结基线正在运行。尚未执行 LoRA 训练和封存 test 评测。
- 已冻结经典离线 Paraformer `v2.0.4` 权重及 SHA-256，并在共享 ASR 环境完成 GPU
  冒烟验证；跨模型实验仅比较 Raw 与 M-WPE，不引入 VAD、标点、语言模型或热词。

## 仓库结构

```text
configs/robust_asr/              冻结实验协议、数据和模型配置
src/robust_asr/
  acoustics/                    阵列几何、RIR 仿真、DRR 与卷积
  dereverb/                     单/多通道 WPE 前端
  models/                       Whisper 推理与 LoRA 配置
  training/                     训练数据、批处理、终端进度与结构化日志
  aishell.py                    AISHELL 数据发现和划分
  baseline.py                   基线实验编排
  manifest.py                   JSONL manifest 与哈希
  scoring.py                    中文 CER 和统计检验
scripts/robust_asr/              可复现实验命令入口
tests/                           鲁棒 ASR 自动测试
docs/robust_asr/                 实验计划、执行记录和项目总结
```

完整职责和本次清理记录见
[代码架构与清理记录](docs/robust_asr/03_代码架构与清理记录.md)。

## 快速验证

```bash
python3 -m venv .venv-robust-asr
./.venv-robust-asr/bin/pip install -e '.[asr,train,evaluation,dev]'
./.venv-robust-asr/bin/python scripts/robust_asr/validate_protocol.py
./.venv-robust-asr/bin/python -m pytest
```

准备数据并生成开发集 RIR：

```bash
export ROBUST_ASR_DATA_ROOT=/path/to/robust_asr
./.venv-robust-asr/bin/python scripts/robust_asr/prepare_aishell.py
./.venv-robust-asr/bin/python scripts/robust_asr/generate_rir_bank.py --split dev
```

在具备 GPU、本地 Whisper 权重和已准备数据的服务器上运行冻结基线：

```bash
./.venv-robust-asr/bin/python scripts/robust_asr/run_frozen_whisper_baseline.py \
  --device cuda --local-files-only
```

Paraformer 跨模型验证复用 Whisper 的 `robust-asr` 环境，并补装 FunASR：

```bash
./.venv-robust-asr/bin/python -m pip install -e \
  '.[asr,train,evaluation,dev,paraformer]'
./.venv-robust-asr/bin/python \
  scripts/robust_asr/run_frozen_paraformer_baseline.py --device cuda
```

## 数据边界

原始语音、RIR、模型权重、训练检查点和实验输出不进入 Git。仓库只保存代码、冻结协议、
测试和结论文档；本地用于开发代码，Linux 服务器保存正式数据并承担 GPU 实验。

## 文档

1. [项目改造与实验计划](docs/robust_asr/00_普通话混响鲁棒ASR改造与实验计划.md)
2. [执行记录与复现](docs/robust_asr/01_执行记录与复现.md)
3. [项目结论与简历面试叙事](docs/robust_asr/02_项目结论与简历面试叙事.md)
4. [代码架构与清理记录](docs/robust_asr/03_代码架构与清理记录.md)
