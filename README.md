# Real-time Speech Enhancement

这是一个面向实时单通道语音前端的工程与实验项目。项目同时保留两类方法：

- 可解释的 `MCRA + Decision-Directed Wiener`；
- official pretrained RNNoise 的实时流式适配。

当前默认听感 Demo 使用 RNNoise R3，但项目结论不是“RNNoise 全场景最优”：它在低 SNR、
非语音型噪声上抑制更强，在 high-SNR 和 babble 条件会过处理目标语音。固定 Whisper
`small.en` 的未见说话人分层 ASR 结果进一步表明，感知降噪改善不等于 WER 改善。

## 当前状态

| 工作 | 状态 |
|---|---|
| VoiceBank+DEMAND 12,396 对音频的数据审计 | 完成 |
| 8-speaker validation 3,224 对增强全量评测 | 完成 |
| official test 824 对增强补充评测 | 完成；历史接触过，不称完全盲测 |
| RNNoise 48/16 kHz persistent-state 流式适配 | 完成 |
| 固定 Whisper v1/v2 管线、缓存和评分 | 完成 |
| ASR development balanced 100 条、四路评测 | 完成；管线验收与故障定位 |
| ASR validation 分层 320 条、四路 v1/v2 | 完成；8 speaker × 10 noise × 4 SNR |
| ASR development/validation full | 未运行；当前停止扩样 |
| VCTK transcript 12,396 条结构映射 | 完成；人工听辨抽查待完成 |

ASR validation 分层 320 条（四路 1,280 输入）上的 v1 corpus WER：

| 输入或策略 | WER |
|---|---:|
| clean | `1.57%` |
| noisy | `5.99%` |
| MCRA + DD-Wiener | `6.58%`；相对 noisy `+0.59 pp`，区间跨零 |
| RNNoise v1 固定解码 | `14.60%`；相对 noisy `+8.61 pp` |
| RNNoise v2 灾难保护 | `11.93%` selective WER；coverage `99.69%` |
| RNNoise/noisy reference oracle | `5.13%`，不可部署 |

RNNoise v1 的 paired bootstrap 95% CI 为 `+5.90` 到 `+11.55 pp`；8/8 个说话人、所有
leave-one-speaker-out、去掉影响最大样本以及 babble/非 babble 分组均保持“比 noisy 更差”。
这已满足当前预设的最小计算停止条件，因此暂不扩到 full。它支持本数据与固定 Whisper
协议下的结论，不外推到所有 ASR 后端。MCRA 的区间跨零，仍只能称“未证明有收益”。
RNNoise/noisy reference oracle 也只把 WER 从 `5.99%` 降到 `5.13%`（最多减少 22 个词错误）；
考虑真实 router 必然低于 oracle 且增加双路推理成本，当前不把重新设计 router 作为近期主线。

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
