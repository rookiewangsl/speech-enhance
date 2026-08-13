# Real-time Speech Enhancement

这是一个面向实时单通道语音前端的工程与实验项目。项目同时保留两类方法：

- 可解释的 `MCRA + Decision-Directed Wiener`；
- official pretrained RNNoise 的实时流式适配。

当前默认听感 Demo 使用 RNNoise R3，但项目结论不是“RNNoise 全场景最优”：它在低 SNR、
非语音型噪声上抑制更强，在 high-SNR 和 babble 条件会过处理目标语音。固定 Whisper
`small.en` 的初步 ASR 结果进一步表明，感知降噪改善不等于 WER 改善。

## 当前状态

| 工作 | 状态 |
|---|---|
| VoiceBank+DEMAND 12,396 对音频的数据审计 | 完成 |
| 8-speaker validation 3,224 对增强全量评测 | 完成 |
| official test 824 对增强补充评测 | 完成；历史接触过，不称完全盲测 |
| RNNoise 48/16 kHz persistent-state 流式适配 | 完成 |
| 固定 Whisper v1/v2 管线、缓存和评分 | 完成 |
| ASR `development_balanced_5` 100 条、四路评测 | 完成 |
| ASR development/validation full | 尚未运行 |
| VCTK transcript 12,396 条结构映射 | 完成；人工听辨抽查待完成 |

ASR balanced 子集上的 corpus WER：

| 输入或策略 | WER |
|---|---:|
| clean | `2.28%` |
| noisy | `5.65%` |
| MCRA + DD-Wiener | `5.51%` |
| RNNoise v1 固定解码 | `40.99%` |
| RNNoise v2 灾难保护 | `14.11%` |
| RNNoise/noisy reference oracle | `4.57%`，不可部署 |

这 100 条只是管线验收和问题定位，不是最终 ASR 主实验结论。下一步应依次运行
development full 和冻结后的 validation full。

## 项目手册

手册压缩为五章：

1. [项目、数据与贡献边界](docs/handbook/01_项目数据与贡献边界.md)
2. [算法架构与实时工程](docs/handbook/02_算法架构与实时工程.md)
3. [代码运行与复现](docs/handbook/03_代码运行与复现.md)
4. [实验结果与 ASR 分析](docs/handbook/04_实验结果与ASR分析.md)
5. [面试问答与公式速查](docs/handbook/05_面试问答与公式速查.md)

总入口见 [项目手册索引](docs/handbook/README.md)。

## 快速开始

```bash
python3 -m venv .venv
./.venv/bin/pip install -e '.[data,dev,evaluation,demo]'
./scripts/setup_rnnoise.sh
./.venv/bin/python -m pytest
```

增强单个 16/48 kHz mono WAV：

```bash
./.venv/bin/python scripts/enhance_rnnoise.py \
  --input input.wav \
  --output output_rnnoise.wav \
  --mode official \
  --compensate-delay
```

真实流式系统必须保留因果延迟；`--compensate-delay` 只用于离线 paired evaluation。
完整数据、Demo、ASR v1/v2 命令见[代码运行与复现](docs/handbook/03_代码运行与复现.md)。

## 主要代码

```text
src/speech_frontend/             音频、STFT、VAD、经典增强和 RNNoise 流式实现
scripts/                         数据、评测、Demo 和 ASR CLI
configs/                         数据源、冻结增强和 ASR 协议
tests/                           自动测试
docs/figures/                    只提交 PNG 文档图
```

原始数据、模型权重、可重建大体积 WAV 和全量 ASR 缓存保留在 T7，不进入 Git。
