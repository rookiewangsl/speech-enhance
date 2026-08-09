# 运行与 Demo 使用说明

最后更新：2026-07-31

本文只维护安装、运行、输出检查和 Demo 评价方法。算法原理见
[`02_项目模块与贡献边界.md`](02_项目模块与贡献边界.md)和
[`03_RNNoise模型架构.md`](03_RNNoise模型架构.md)，冻结结果见
[`06_结果与结论说明.md`](06_结果与结论说明.md)。

## 1. 项目路径与运行环境

项目目录：

```bash
cd /Users/shilongwang/Library/CloudStorage/Dropbox/Code/cs61b_sp25/projects/realtime_speech_enhancement
```

首次准备环境：

```bash
python3 -m venv .venv
./.venv/bin/pip install -e '.[data,dev,evaluation,demo]'
./scripts/setup_rnnoise.sh
./.venv/bin/python -m pytest
```

各步骤的作用：

| 命令 | 作用 |
|---|---|
| `python3 -m venv .venv` | 创建项目独立 Python 环境 |
| `pip install -e ...` | 安装项目、数据、评价、测试和麦克风依赖 |
| `setup_rnnoise.sh` | 获取固定 RNNoise 源码/模型并构建共享库 |
| `pytest` | 验证音频契约、DSP、VAD、RNNoise、重采样和 controller |

正常验收应看到全部自动测试通过。若只补装麦克风依赖：

```bash
./.venv/bin/pip install -e '.[demo]'
```

## 2. 已有音频：正式 RNNoise 推理

### 2.1 输入要求

- WAV；
- 单声道；
- 16 kHz 或 48 kHz；
- 项目读取后转换成 `[-1, 1]` normalized float；
- 16 kHz 输入在内部经过状态化 16→48→16 kHz 适配；
- 48 kHz 输入直接进入 RNNoise 核心。

### 2.2 推荐命令

```bash
./.venv/bin/python scripts/enhance_rnnoise.py \
  --input input.wav \
  --output output_rnnoise.wav \
  --mode official \
  --compensate-delay \
  --vad-json output_vad.json
```

参数解释：

| 参数 | 含义 |
|---|---|
| `--mode official` | 当前正式默认 R3 |
| `--compensate-delay` | 离线 A/B 和 paired metric 时前移固定延迟 |
| `--vad-json` | 保存每个 RNNoise 帧的 VAD probability |
| `--chunk-size N` | 用任意 chunk 大小模拟上层流式输入 |
| `--pcm-compatible` | 使用与官方 PCM 路径一致的量化方式做复现检查 |

`conservative`和`aggressive`是已经被听感否决的 R4 消融，只用于复现实验，不应作为正式
Demo 默认输出。

### 2.3 为什么离线比较要补偿延迟

16 kHz链路的固定延迟为：

```text
RNNoise core：160 samples
两段 FIR：     42 samples
总计：        202 samples = 12.625 ms
```

不补偿延迟时，增强语音与 clean reference 错位，SI-SDR等逐样本指标会严重失真。实时系统
不能消除因果延迟，只能向下游报告timestamp offset；`--compensate-delay`用于离线输出和
评价对齐。

## 3. 固定离线对比 Demo

项目固定失败样本：

```bash
./.venv/bin/python scripts/run_demo.py \
  --input data/processed/voicebank/noisy/p232_005.wav \
  --clean data/processed/voicebank/clean/p232_005.wav \
  --output-dir outputs/final/demo_p232_005 \
  --chunk-size 137
```

`137`不是16 kHz下10 ms帧长`160`的整数倍，用于证明wrapper可以接受真实应用中的任意
chunk，而不是只能处理预先切好的完整帧。

输出：

```text
outputs/final/demo_p232_005/
  clean_reference.wav
  noisy_input.wav
  r3_official.wav
  c1_continuity_probe.wav
  comparison.png
  report.json
```

推荐试听顺序：

1. `clean_reference.wav`：确认目标内容；
2. `noisy_input.wav`：识别前半段结构化金属噪声；
3. `r3_official.wav`：听金属噪声是否下降，以及语音是否断续；
4. `c1_continuity_probe.wav`：检查断续是否改善、是否新增泵动；
5. 回到`noisy_input.wav`，避免听觉适应后忘记原始噪声。

macOS命令行试听：

```bash
afplay outputs/final/demo_p232_005/clean_reference.wav
afplay outputs/final/demo_p232_005/noisy_input.wav
afplay outputs/final/demo_p232_005/r3_official.wav
afplay outputs/final/demo_p232_005/c1_continuity_probe.wav
```

`comparison.png`用于观察waveform、noisy/R3 spectrogram和VAD；`report.json`记录采样率、
chunk、延迟、RTF、VAD和paired metrics。

## 4. MacBook麦克风实时采集 Demo

### 4.1 Demo处理方式

```text
CoreAudio microphone
→ 48 kHz mono float32
→ 480 samples / 10 ms callback blocks
→ queue
→ persistent RNNoise state
→ 保存raw/enhanced WAV和report
```

处理在录音期间实时进行，但不会同时从扬声器外放。这样可以避免没有AEC时形成反馈和额外
回声，也能让Demo聚焦于capture、block、state、RTF和输出正确性。

### 4.2 先枚举设备

```bash
./.venv/bin/python scripts/realtime_mic_demo.py \
  --output-dir outputs/mic_demo \
  --list-devices
```

选择`in > 0`的输入设备。设备编号会在插拔显示器、耳机、USB麦克风后变化，不能永久写死。

2026-07-31本机曾显示：

```text
0 HDMI                    0 input
1 P275MV                  0 input
2 MacBook Air Microphone  1 input
3 MacBook Air Speakers    0 input
4 rookie Microphone       1 input
```

这次检查中内置麦克风是`2`，以后仍应以最新列表为准。

### 4.3 macOS麦克风权限

首次运行时检查：

```text
System Settings
→ Privacy & Security
→ Microphone
→ 允许 Codex 或 Terminal
```

修改权限后若仍无数据，完全退出并重新打开发起命令的应用。

### 4.4 录制10秒

以下`2`必须替换成最新设备列表中的MacBook输入编号：

```bash
./.venv/bin/python scripts/realtime_mic_demo.py \
  --output-dir outputs/mic_demo \
  --duration 10 \
  --device 2
```

运行中每0.5秒打印一次最近的RNNoise VAD probability。完成后得到：

```text
outputs/mic_demo/
  microphone_raw.wav
  microphone_rnnoise.wav
  report.json
```

试听：

```bash
afplay outputs/mic_demo/microphone_raw.wav
afplay outputs/mic_demo/microphone_rnnoise.wav
```

保存的增强文件保留真实10 ms因果延迟。顺序试听通常不受影响；若做逐样本波形比较，应固定
前移480个48 kHz samples，而不是针对每个录音用参考信号搜索“最佳lag”。

## 5. 推荐麦克风测试脚本

一次10–15秒录音可按固定顺序完成：

| 时间 | 操作 | 观察点 |
|---|---|---|
| 0–2 s | 不说话 | noise floor是否下降，是否产生调制噪声 |
| 2–6 s | 正常音量读固定文本 | 是否吞字、断续、变闷 |
| 6–8 s | 再次静音 | VAD是否下降，残余噪声是否稳定 |
| 8–12 s | 敲键盘/开风扇并重复文本 | 瞬态或稳定噪声抑制、语音保护 |

建议保持：

- 麦克风距离和说话音量不变；
- raw和enhanced来自同一次录音，不要分别录两次；
- 多种算法使用同一份raw输入；
- 播放咖啡厅噪声时把扬声器放远，或用隔离良好的第二设备；
- 当前项目没有AEC，不要把扬声器回声误当成普通环境噪声问题。

## 6. 如何评价麦克风 Demo

麦克风录音没有同步clean reference，所以不应计算SI-SDR或STOI。使用以下检查。

### 6.1 `report.json`硬检查

正常结果应满足：

| 字段 | 期望 |
|---|---|
| `input_device` | 选中的输入设备名称 |
| `sample_rate` | `48000` |
| `block_samples` | `480` |
| `captured_samples` | 10秒约`480000` |
| `algorithmic_delay_samples` | `480` |
| `algorithmic_delay_ms` | `10` |
| `processing_rtf` | `< 1` |
| `dropped_blocks` | `0` |
| `callback_status` | 空列表 |
| `output_clipping_samples` | `0` |

`input_peak`过低可能表示设备/输入增益错误，接近`1.0`且大量clipping则表示输入增益过高。

### 6.2 主观A/B表

每个场景记录：

| 场景 | 背景噪声 | 吞字 | 断续 | 金属感 | 泵动 | 音色 | 偏好 |
|---|---|---|---|---|---|---|---|
| 安静 |  |  |  |  |  |  |  |
| 风扇 |  |  |  |  |  |  |  |
| 键盘 |  |  |  |  |  |  |  |
| 咖啡厅 |  |  |  |  |  |  |  |
| 音乐 |  |  |  |  |  |  |  |
| 第二说话人 |  |  |  |  |  |  |  |

不要只问“噪声是否变小”，还要检查语音完整性、自然度和局部强伪影。

## 7. 常见问题

### 找不到RNNoise共享库

```bash
./scripts/setup_rnnoise.sh
```

### 找不到`sounddevice`

```bash
./.venv/bin/pip install -e '.[demo]'
```

### 设备没有输入通道

重新运行`--list-devices`，选择`in > 0`设备。HDMI和扬声器通常只有output。

### `Invalid sample rate`

在Audio MIDI Setup检查设备是否支持48 kHz，优先使用MacBook内置麦克风。

### 两秒内没有audio frames

检查设备编号、macOS权限、系统输入电平；脚本会给出明确的timeout提示。

### `dropped_blocks > 0`

关闭高负载程序，重新运行并检查RTF。平均RTF小于1仍不能完全排除调度抖动。

### 输出比输入稍晚

48 kHz RNNoise核心有一帧约10 ms因果延迟，这是算法属性，不是文件损坏。

## 8. Demo展示顺序

三分钟展示建议：

1. 说明项目位于音频前端的single-channel noise suppression位置；
2. 播放固定样本noisy与R3；
3. 用图展示频谱变化、VAD、12.625 ms延迟和RTF；
4. 解释R4为何客观指标较好但被听感否决；
5. 运行或展示MacBook麦克风Demo；
6. 明确官方模型与自主DSP/实时工程的贡献边界。

## 9. 完整数据流程复现

在项目根目录执行：

```bash
# 迁移目录后若虚拟环境仍指向旧路径
./.venv/bin/python -m pip install --no-deps -e .

# 下载完整 28-speaker train
./.venv/bin/python scripts/download_voicebank.py \
  --data-root data --subset train28

# 20/8/official-test 预处理
./.venv/bin/python scripts/prepare_voicebank.py \
  --data-root data --protocol full28 \
  --development-speakers 20 --seed 20260724 --workers 4

# 校验 speaker/utterance 无泄漏
./.venv/bin/python scripts/audit_full_protocol.py \
  --project-root . --manifest-root data/manifests \
  --output outputs/full_protocol/data_audit.json
```

冻结评估示例：

```bash
# RNNoise，完整未见 speaker validation
./.venv/bin/python scripts/evaluate_rnnoise.py \
  --manifest data/manifests/validation.jsonl \
  --project-root . \
  --output-root outputs/full_protocol/rnnoise/validation_full \
  --chunk-size 137

# 保守 DD-Wiener
./.venv/bin/python scripts/evaluate_enhancement.py \
  --manifest data/manifests/validation.jsonl \
  --project-root . \
  --output-root outputs/full_protocol/classical/validation_full \
  --alpha-dd 0.92 --gain-floor 0.20 \
  --methods mcra_dd_wiener

# 按 noise/SNR 汇总
./.venv/bin/python scripts/summarize_metrics_by_condition.py \
  --input outputs/full_protocol/rnnoise/validation_full/metrics/rnnoise_metrics.csv \
  --output outputs/full_protocol/rnnoise/validation_full/metrics/by_condition.json
```

固定 Demo 的试听顺序仍是 `clean reference → noisy input → R3 → C1`。完整数据结果不要求播放
3,224 条音频；面试时增加一页 condition 表，主动说明 high-SNR 和 babble 失败即可。
