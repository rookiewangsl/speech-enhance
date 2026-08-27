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
- 已实现 Raw、单通道 WPE（10/40 taps）和多通道 WPE（10 taps）四条前端分支。
- 已实现普通话文本归一化、CER、替换/删除/插入统计和 paired bootstrap。
- 已验证冻结 Whisper GPU 推理和 LoRA 前向/反向链路。
- 已完成协议校验、数据泄漏检查和自动测试。
- 尚未运行正式冻结基线、WPE 消融和 LoRA 训练。

## 仓库结构

```text
configs/robust_asr/              冻结实验协议、数据和模型配置
src/robust_asr/
  acoustics/                    阵列几何、RIR 仿真、DRR 与卷积
  dereverb/                     单/多通道 WPE 前端
  models/                       Whisper 推理与 LoRA 配置
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

## 数据边界

原始语音、RIR、模型权重、训练检查点和实验输出不进入 Git。仓库只保存代码、冻结协议、
测试和结论文档；本地用于开发代码，Linux 服务器保存正式数据并承担 GPU 实验。

## 文档

1. [项目改造与实验计划](docs/robust_asr/00_普通话混响鲁棒ASR改造与实验计划.md)
2. [执行记录与复现](docs/robust_asr/01_执行记录与复现.md)
3. [项目结论与简历面试叙事](docs/robust_asr/02_项目结论与简历面试叙事.md)
4. [代码架构与清理记录](docs/robust_asr/03_代码架构与清理记录.md)
