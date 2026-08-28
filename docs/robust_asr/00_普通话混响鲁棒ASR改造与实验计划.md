# 普通话混响鲁棒 ASR：项目改造与实验计划

最后更新：2026-08-28
状态：方案 v0.3；AISHELL-1、开发集 RIR、Linux GPU 环境、LoRA 反向链路和正式 Raw 开发集
基线已经验收，WPE 开发集消融正在运行。训练 Dataset/Collator 已实现；train/test RIR、优化循环、
LoRA 训练与 test 消融尚未执行。除[执行记录](01_执行记录与复现.md)明确列出的结果外，
本文中的实验规模和性能数值均为计划，不是已经得到的结果。
项目结论、简历 bullet 和面试叙事集中维护在[项目总结文档](02_项目结论与简历面试叙事.md)。

## 0. 当前实施状态

截至 2026-08-26，已完成：

- `configs/robust_asr/*.json` 六份冻结配置及跨文件 SHA/一致性校验；
- 中文 NFKC/字符过滤 normalizer、CER、S/D/I 和 paired bootstrap；
- JSONL 原子写入、speaker/room 分组泄漏检查、MCT/RIR 确定性采样；
- 四麦 UCA 房间几何采样、RT60 容差、DRR、完整卷积、共同增益和削波保护；
- offline WPE 的 NumPy reference smoke 后端及 NARA-WPE 正式后端接口；
- Whisper LoRA 精确目标层筛选、参数计数检查和 4070 训练预算回退逻辑；
- 3 模型 × 4 前端 × 5 RT60 的 60-cell/60,000-input 正式矩阵生成；
- 协议校验、矩阵导出和合成四通道 WPE 单元测试；
- AISHELL-1 主包与资源包 SHA 校验、400 个 speaker 压缩包安全解包、141,925 WAV 全量审计；
- 120,098 条训练、14,326 条开发、7,176 条测试可用清单；325 条无官方转写 WAV 均有审计记录且
  不进入训练或评测；
- 可复现的 20 小时 train、1,000 条 dev-model、500 条 dev-frontend、1,000 条 test-reverb 和
  500 条 test-measured-RIR 子集；
- 固定 revision 的 `openai/whisper-small` CPU 开发集烟雾运行，覆盖 clean、三档 RT60 与四路前端；
- Linux RTX 4070 上的本地权重加载、真实语音 GPU 推理与 LoRA forward/backward 验收；
- 500 条开发集上的 Clean+Raw 五档 RT60 正式基线、成对置信区间与条件化 DRR 分析；
- Clean/MCT 训练 Dataset、epoch 级确定性 train RIR 采样、文件校验与 Whisper batch collator；
- 正式 test RIR 的同几何 family 跨 RT60 配对生成逻辑及回归测试；
- 活动代码已收敛为单一 `robust_asr` 包；历史实时增强代码从当前代码树移除。

当前 reference WPE 只用于验证数学接口、shape、缓存和测试，不得作为正式 WPE 结果。正式实验仍
要求 NARA-WPE。繁体转简体、Pyroomacoustics、Whisper/PEFT 也采用延迟导入，缺依赖时明确失败。

无数据阶段可运行：

```bash
./.venv/bin/python scripts/robust_asr/validate_protocol.py
./.venv/bin/python scripts/robust_asr/build_experiment_matrix.py --utterances 10
./.venv-robust-asr/bin/python -m pytest
```

仍待执行：完成全量 NARA-WPE 开发集消融、train/test RIR bank、优化循环与 checkpoint/dev 选择、
训练 CLI、LoRA 训练和 test 推理。Linux + RTX 4070 环境统一安装 `.[asr,train,evaluation,dev]`。

## 1. 项目定位

项目从“固定 ASR 后端前的实时单通道降噪”改造为：

> 面向普通话远场混响的鲁棒 ASR：使用可控的多通道 RIR 构造混响语音，比较单通道与
> 多通道 WPE 去混响，并通过 Whisper LoRA 多条件训练研究固定前端与模型适配的互补性。

校招能力映射如下：

| 能力 | 本项目证据 |
|---|---|
| 数据工程 | AISHELL-1 下载校验、20 小时可复现子集、RIR bank、动态增强、manifest 和泄漏审计 |
| 声学建模 | RT60、DRR、房间/阵列几何、多通道卷积和条件化分析 |
| 传统语音前端 | 单通道/多通道 WPE、参数与容量公平消融、失败条件分析 |
| ASR 模型适配 | Whisper-small LoRA、插入位置消融、clean 与 multi-condition 公平训练 |
| ASR 评测 | 中文规范化 CER、S/D/I、paired bootstrap、RT60 主分析与 DRR 条件化分析 |
| 科研方法 | 预注册假设、受控变量、room-disjoint 测试、负结果与结论边界 |

项目不以部署为目标，不实现在线 WPE、流式 Whisper、ONNX、量化、端侧适配或实时延迟优化。

## 2. 与原项目及科研经历的关系

### 2.1 与原英文降噪项目

原英文单通道降噪项目不再保留在活动代码树中，也不进入新项目实验矩阵。清理前代码和结果可从
Git 提交 `767413b6df696f2b99aa5e5b1d52769834520b9e` 审计或恢复，但不应重新引入当前主线。
新实现继承的只是以下工程方法：

- 音频读写、采样率、有限值和幅值检查；
- manifest、审计和确定性抽样模式；
- 错误分解和 paired bootstrap；
- 实验协议冻结、结果汇总和异常诊断思路。

新项目不得继承以下假设：

- VoiceBank 的 speaker/noise/SNR 字段；
- 英文 `small.en` 模型和英文 normalizer；
- 单通道实时、VAD、RNNoise 延迟和 RTF 作为主叙事；
- “感知降噪等于 ASR 改善”的假设。

### 2.2 与水下声学科研

科研工作利用目标运动特性，在到达角—多普勒域中进行 STAP/MVDR 二维滤波；语音项目使用
麦克风—STFT 历史帧上的多通道长时线性预测。两者算法和物理假设不同：普通静止说话人没有
可利用的目标多普勒维度，因此不直接移植原水声算法。

可迁移能力是：

```text
多通道观测
→ 构造空时统计量
→ 估计相关/协方差矩阵
→ 求解自适应滤波器
→ 抑制不希望的传播成分
```

准确表述为：项目把多通道阵列、统计建模和空时自适应处理能力迁移到房间语音去混响，而不是
宣称 WPE 等同于水声 STAP/MVDR。

## 3. 已冻结的范围与决策

| 项目 | 决策 |
|---|---|
| 语言 | 只研究普通话 |
| 干净语音 | AISHELL-1 |
| ASR 主模型 | `openai/whisper-small` multilingual |
| 训练方法 | LoRA，不做全参数微调或 QLoRA |
| 目标场景 | 中小型办公室/会议室、桌面阵列、单个静止说话人 |
| 混响 | 主实验使用 Pyroomacoustics 可控仿真 RIR |
| 外部测试 | 核心完成后增加少量未见实测 RIR |
| 麦克风 | 四通道水平面均匀圆阵，半径 5 cm，Mic 0 为参考通道 |
| 干扰 | 第一阶段只加混响，不加噪声、重叠说话或运动 |
| 去混响 | offline WPE；Raw、S-WPE-10、S-WPE-40、M-WPE-10 |
| 训练规模 | 固定 20 小时 AISHELL-1 子集；超时才统一降为 10 小时 |
| 主要指标 | 中文规范化 CER；信号指标只作解释 |
| Paraformer | 暂不微调；冻结跨模型验证为可选扩展 |
| AISHELL-4 | 当前不做 |
| 部署 | 当前不做 |

## 4. 研究问题与预注册假设

### 4.1 主要问题

1. 冻结 Whisper 的 Raw CER 如何随实测 RT60 变化？同一 RT60 档内的 CER 差异能否由 DRR 解释？
2. WPE 是否降低 CER，还是只改善信号指标并引入 ASR 不喜欢的失真？
3. M-WPE 是否优于相同时间跨度或相近预测维数的 S-WPE？
4. clean-only LoRA 的收益来自普通话领域适配，还是 multi-condition LoRA 才能提高混响鲁棒性？
5. WPE 与 multi-condition LoRA 是互补、冗余，还是存在前端—模型失配？
6. 仿真训练得到的方向性结论能否迁移到少量实测 RIR？

### 4.2 假设

- H1：Raw CER 随 RT60 增大而整体上升，且 deletion/insertion 在重混响下增多。
- H2：WPE 在中重度混响下比轻度混响更可能改善 CER。
- H3：M-WPE-10 优于 S-WPE-10；若也优于 S-WPE-40，才能更有力归因于多通道观测。
- H4：MCT-LoRA 在 raw reverb 上优于 Clean-LoRA，同时 clean CER 不发生不可接受退化。
- H5：WPE 与 MCT-LoRA 的增益不保证相加；训练后前端收益可能下降，形成冗余。
- H6：信号级去混响指标与 CER 改善不完全一致，前端选择最终由 dev CER 决定。

上述均是假设而非目标答案。即使 WPE 没有改善 CER，只要实验协议正确、原因分析充分，仍是有效
项目结论，不得筛掉负结果。

## 5. 总体系统与网络

### 5.1 数据和推理路径

```text
AISHELL-1 mono clean speech
        │
        ├── clean ───────────────────────────────────────────────┐
        │                                                        │
        └── 与四条 source→mic RIR 卷积                           │
                 │                                               │
                 ├── y0 ─────────────── Raw reference ───────────┤
                 ├── y0 ─────────────── S-WPE-10/40 ─────────────┤
                 └── [y0,y1,y2,y3] ─── M-WPE-10 → Mic 0 ─────────┤
                                                                  ↓
                                   Whisper-small / LoRA adapter
                                                                  ↓
                                         hypothesis → normalized CER
```

RIR 只用于造数据、分层和诊断；WPE 不读取真实 RIR、RT60 或干净语音，属于盲去混响。

### 5.2 Whisper-small

采用 Hugging Face `openai/whisper-small` multilingual，约 2.44 亿参数。核心路径为：

```text
16 kHz waveform
→ 80-bin log-Mel
→ 2-layer convolutional frontend
→ 12-layer Transformer encoder, width 768
→ 12-layer autoregressive Transformer decoder, width 768
→ multilingual tokenizer
```

训练和推理固定 `language=zh`、`task=transcribe`、不输出 timestamp。主实验不外接语言模型。

### 5.3 LoRA 插入位置

短程试验比较：

```text
P1 Encoder-QV:
  encoder.layers.*.self_attn.{q_proj,v_proj}

P2 Encoder+Decoder-QV:
  encoder.layers.*.self_attn.{q_proj,v_proj}
  decoder.layers.*.self_attn.{q_proj,v_proj}
  decoder.layers.*.encoder_attn.{q_proj,v_proj}
```

第一阶段不训练 `k_proj/out_proj/fc1/fc2/conv/lm_head/embedding`。基础模型全部冻结，启动时
打印并审计可训练参数名、数量和占比。

## 6. 仓库和产物组织

当前活动代码：

```text
src/robust_asr/
  acoustics/            RIR 生成、RT60/DRR、卷积和幅值协议
  dereverb/             NARA-WPE 封装和固定前端
  models/               Whisper 冻结推理和 LoRA 注入
  training/             Clean/MCT 数据、Whisper 批处理与训练日志
  aishell.py             AISHELL 审计和确定性切分
  manifest.py            JSONL manifest 与哈希
  scoring.py             CER、S/D/I、bootstrap 和条件汇总
  baseline.py            冻结基线编排

scripts/robust_asr/
  download_aishell.py
  extract_aishell.py
  prepare_aishell.py
  generate_rir_bank.py
  build_experiment_matrix.py
  validate_protocol.py
  run_frozen_whisper_baseline.py

configs/robust_asr/
  data.json
  rir.json
  wpe.json
  whisper.json
  lora.json
  evaluation.json

tests/
docs/robust_asr/
```

优化循环、checkpoint 选择、训练 CLI 和正式结果绘图尚未实现；它们是下一阶段新增项，不应在现状
清单中伪装成已有文件。详细职责见[代码架构与清理记录](03_代码架构与清理记录.md)。

新代码只进入 `robust_asr` 命名空间，不恢复旧项目模块。

大文件由任务专用变量指定，代码不得硬编码某块磁盘：

```bash
export ROBUST_ASR_DATA_ROOT=/path/to/robust_asr
```

Git 只保存代码、配置、可审计 manifest、小型 CSV/JSON 汇总和 PNG 图；AISHELL 压缩包、WAV、
RIR bank、模型权重、增强语音、特征和全量推理缓存留在数据根目录。

## 7. 数据处理计划

### 7.1 AISHELL-1 获取和审计

来源使用 [OpenSLR SLR33](https://www.openslr.org/33/)。记录下载 URL、文件大小、SHA-256、
解压文件数、总时长、speaker/utterance 数和官方 split。基础检查：

- WAV 可读取、单通道、16 kHz、有限值；
- transcript 非空且 utterance ID 唯一；
- 音频与 transcript 一一对应；
- 官方 train/dev/test speaker 不交叉；
- 时长、峰值、RMS 和异常短/长语句分布可审计；
- clean 加 RIR 尾部后不超过 Whisper 30 秒限制。

### 7.2 20 小时训练子集

从官方 train 抽取固定 20 小时：

- 尽可能覆盖全部 train speaker，而不是只选少数完整说话人；
- 每位说话人的抽样时长尽量接近；
- 保持性别、语句长度和五个文本领域的基本分布；
- seed 固定为 `2026`；
- 保存候选集 SHA、抽样脚本版本、最终 manifest SHA 和审计报告。

排除规则：`duration < 0.5 s`、clean `duration > 25 s`、损坏、非有限值、严重削波、空转写或
重复 ID。所有排除项必须带原因写入审计文件。

### 7.3 Dev/Test 子集

| 子集 | 来源 | 数量 | 用途 |
|---|---|---:|---|
| `dev_model` | official dev | 1,000 | LoRA loss、checkpoint 和位置选择 |
| `dev_frontend` | official dev，和上者互斥 | 500 | WPE 参数检查和前端选择 |
| `test_clean_full` | official test 全量 | 全量 | 三个模型的 clean CER |
| `test_reverb_fixed` | official test | 1,000 | 五档 RT60 × 四种前端 × 三个模型 |
| `test_measured_rir` | official test 子集 | 300–500 | 最终仿真外测试 |

所有子集覆盖尽可能多的官方 speaker，并保存确定性抽样 manifest。

### 7.4 文本规范化与 CER

主指标 normalizer 固定为：

1. Unicode NFKC；
2. 繁体转简体；
3. 拉丁字母转小写；
4. 删除空格、标点和非语言符号；
5. 保留汉字、拉丁字母和数字；
6. 不自动把阿拉伯数字转换为中文读法。

同时保存 raw hypothesis、normalized hypothesis、strict CER（不做繁简转换）和 S/D/I。为
normalizer 建立金标准单元测试，避免库升级静默改变指标。

## 8. RIR 生成协议

### 8.1 仿真器

使用 [Pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html)：

- 3D ShoeBox；
- Image Source Method；
- `fs=16000`；
- `use_rand_ism=true`，降低规则镜像源产生的 sweeping echo；
- 启用 air absorption；
- 第一版使用频率无关的等效墙面吸收，以便控制 RT60；
- 保存 Pyroomacoustics、NumPy、SciPy 版本和完整配置。

### 8.2 房间、阵列和声源

| 参数 | 分布/固定值 |
|---|---|
| room length | Uniform `[4, 8] m` |
| room width | Uniform `[3, 6] m` |
| room height | Uniform `[2.5, 3.5] m` |
| array | 4-mic horizontal UCA |
| array radius | `0.05 m` |
| array center height | `0.8 m` |
| reference channel | Mic 0 |
| source height | Uniform `[1.3, 1.8] m` |
| source-array distance | Uniform `[1, 4] m` |
| azimuth | Uniform `[0, 360°)` |
| wall margin | source/array 至少 `0.5 m` |
| source/mic directivity | omnidirectional |
| source motion | static |
| additive noise | none |

几何采样必须先验证全部点位于房间内，并拒绝距离、墙距或阵列边界不满足约束的样本。

保留 `1–4 m` 的声源—阵列距离变化，用于覆盖近讲到远场条件并丰富 DRR 分布。距离不作为
独立的主要实验轴，只作为几何元数据和辅助协变量；主分析仍是 Raw CER–RT60，DRR 用于解释同一
RT60 档内的差异。由于卷积后会统一归一化到目标 RMS，本阶段距离改变的是直达声/混响声结构，而不是
通过音量衰减制造 SNR 变化。

距离变化不是 RIR 生成的数学必需条件，但对本项目有必要：RT60 只描述房间整体衰减速度，不能唯一
决定某个麦克风位置的混响严重程度。在相同 RT60 下，不同距离和摆位会改变直达声与反射声的相对能量、
反射到达时延和多通道空间结构，从而生成不同 DRR 和 CER。固定 `2 m` 会缩窄 DRR 及场景覆盖，不利于
开展“同一 RT60 档内为什么 CER 不同”的分析，因此不采用。

当前仿真不包含办公室、会议室、教室等语义房型，全部为尺寸连续变化的 3D 矩形 ShoeBox 房间；
墙面使用频率无关的等效吸收率，不模拟家具、门窗、非矩形结构或频率相关材料。因此对外表述应为
“多个未见矩形房间几何上的可控仿真”，不宣称覆盖多种真实房型。

### 8.3 RT60 与 RIR bank

| Split | 几何 | 位置/RT60 | RIR 规模 |
|---|---:|---|---:|
| Train | 100 个 room | 每 room 约 10 个位置，RT60 连续采样 `[0.2,1.0] s` | 约 1,000 组四通道 |
| Dev（已生成） | 20 个未见 room | 每档 RT60 独立采样 2 个位置 | 200 组 |
| Test（计划） | 20 个未见 room | 3 个几何 family × `0.2/0.4/0.6/0.8/1.0 s` | 约 300 组 |

生成后测量每个通道实际 RT60。当前生成器使用四通道实测 RT60 的中位数进行校准，必须满足：

\[
|RT60_{measured}-RT60_{target}|\leq \max(0.05\text{s},0.1RT60_{target}).
\]

生成器最多迭代五次调整设计 RT60；仍不满足时该 RIR 不得进入 bank。五档实验条件按目标 RT60
分组，同时报告各档四通道实测值；Mic 0 作为 Raw ASR 参考通道，其他通道实测值全部保存。

Train/dev/test 的 room ID、几何模板和生成 seed 必须完全隔离。待生成的正式 test bank 使用
RIR family：不同 family 之间的距离仍在 `1–4 m` 内随机，同一 family 内固定房间、阵列、声源位置、
距离和方位角，只生成五个不同 RT60 版本。这样同时保留距离/DRR 多样性与 RT60 严格配对性。

已生成的 `pyroom_v1/dev` 200 组 RIR 沿用旧生成器：20 个房间×五档 RT60×每档两个独立几何位置。
它的各 RT60 档几何来自同一分布，但不是逐条配对。该 bank 可继续用于开发集平均 Raw CER–RT60 曲线、DRR
条件化分析和 WPE 选型，无需因 `1–4 m` 距离变化而重新生成；但不得将不同 RT60 的两条 RIR 宣称为
“相同几何仅改变吸收率”的配对样本。

### 8.4 DRR 和元数据

对仿真 RIR 额外计算 direct-path-only 分量，用其与剩余反射能量计算 DRR：

\[
DRR=10\log_{10}\frac{E_{direct}}{E_{full-direct}}.
\]

每组 RIR 至少记录：

```text
rir_id, rir_family_id, split, room_id
room_dimensions, target_rt60, measured_rt60_per_channel
source_position, array_center, microphone_positions
distance, azimuth, drr_per_channel
sample_rate, simulator_version, config_sha, random_seed
rir_length, peak, rms, generation_status
```

### 8.5 卷积、幅值和尾部

- 使用完整线性卷积，输出长度为 `N_speech + N_rir - 1`；
- 不在原语音结束处截断 RIR tail；
- 四通道使用同一个增益，禁止逐通道归一化；
- 以 raw Mic 0 为准把整组混响信号调整到约 `-25 dBFS RMS`；
- 若任一通道可能削波，四通道共同缩小并保留约 1 dB peak headroom；
- WPE 输出不单独改变增益，只做有限值和削波保护；
- clean baseline 使用相同目标 RMS 规则；
- train RIR 由 `hash(utt_id, epoch, seed)` 确定性抽取。

## 9. WPE 前端方案

使用 [NARA-WPE](https://github.com/fgnt/nara_wpe) 的离线迭代算法。固定参数：

```yaml
sample_rate: 16000
n_fft: 512
win_length: 512
hop_length: 128
window: hann
delay: 3
iterations: 3
psd_context: 0
statistics_mode: full
dtype: complex128
reference_channel: 0
```

前端条件：

| ID | 输入 | taps | 预测向量维数 | 目的 |
|---|---|---:|---:|---|
| F0 Raw | Mic 0 | — | — | 混响 ASR 基线 |
| F1 S-WPE-10 | Mic 0 | 10 | 10 | 与 M-WPE 保持相同时间跨度 |
| F2 S-WPE-40 | Mic 0 | 40 | 40 | 与 M-WPE 保持相近预测维数 |
| F3 M-WPE-10 | Mic 0–3 | 10 | 40 | 多通道主方案，最后取 Mic 0 |

所有 RT60 使用同一套参数，不允许针对 test 档位分别调参。先在 `dev_frontend` 运行默认值；只有
当 M-WPE 在 `RT60>=0.6 s` 明显恶化时，才在一个 200 条 dev 诊断子集搜索：

```text
delay ∈ {2, 3, 4}
M-WPE taps ∈ {5, 10, 20}
iterations 固定 3；仍不收敛才比较 5
```

参数冻结后 test 只运行一次。每条输出检查 shape、长度、同步、NaN/Inf、峰值、RMS、求解器
warning 和处理时间；Mic 0 固定，不动态选择“最好通道”。

## 10. Whisper 与 LoRA 训练方案

### 10.1 固定解码

主实验：

```yaml
language: zh
task: transcribe
return_timestamps: false
temperature: 0
do_sample: false
num_beams: 1
condition_on_prev_tokens: false
```

greedy decoding 用于所有主表，保证确定性和计算可控。最终最佳模型可额外报告 beam 5，但不能
和主表的 greedy 结果混用。

### 10.2 LoRA 固定参数

```yaml
r: 8
lora_alpha: 16
lora_dropout: 0.05
bias: none
peft_wrapper: generic_for_whisper_input_features
task_type: null
```

Whisper 是语音条件生成模型，encoder 接收 `input_features` 而非文本 `input_ids`。因此不使用
PEFT 的文本 `PeftModelForSeq2SeqLM` 包装器，而使用通用透传包装器；LoRA 目标层、rank、alpha、
dropout、adapter 保存格式和可训练参数语义不变。正式训练前必须使用真实 `input_features` 和 labels
完成一次 FP16 forward/backward 验收，不能只检查 adapter 数量和参数冻结。

不做 rank 大网格、DoRA、QLoRA、all-linear、全参数微调或额外语言模型。`r=16` 只在训练损失和
dev CER 均显示欠拟合时作为条件触发的诊断。

### 10.3 插入位置短程消融

| ID | 数据 | 位置 | 训练量 |
|---|---|---|---:|
| LP1 | 5 h；50% clean + 50% raw reverb | Encoder-QV | 500 optimizer steps |
| LP2 | 同一数据、seed 和顺序 | Encoder+Decoder-QV | 500 optimizer steps |

选择指标：

\[
CER_{robust}=mean(CER_{0.4},CER_{0.6},CER_{0.8},CER_{1.0}),
\]

且 clean CER 不得明显恶化。若二者差异很小，选择参数更少的 Encoder-QV。位置只在 dev 上选，
test 不参与。

### 10.4 正式训练

位置冻结后只训练两个正式 adapter：

| ID | 数据 | 每 epoch 样本量 | 目的 |
|---|---|---:|---|
| L1 Clean-LoRA | 20 h clean | 固定 | 普通话领域适配控制组 |
| L2 MCT-LoRA | 50% clean + 50% raw reverb | 与 L1 相同 | 混响多条件训练主模型 |

MCT 不是把 20 h clean 与 20 h reverb 拼成 40 h，而是每次样本读取按 50/50 选择形式，使 L1/L2
的样本数、optimizer steps 和训练预算一致。第一阶段不把 WPE 输出加入训练，以便测试固定前端
与 raw-reverb 模型适配的互补性。

### 10.5 优化参数与算力闸门

```yaml
precision: fp16
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
effective_batch_size: 16
dataloader_num_workers: 16
dataloader_prefetch_factor: 4
optimizer: AdamW
learning_rate: 1.0e-4
weight_decay: 0.01
warmup_ratio: 0.05
lr_scheduler: linear
max_epochs: 3
gradient_checkpointing: true
max_grad_norm: 1.0
seed: 2026
logging:
  console_interval_steps: 20
  structured_interval_steps: 10
  selection_metric: dev_reverb_cer
  heavy_rt60_seconds: [0.8, 1.0]
```

batch 2 OOM 时改成 batch 1、accumulation 16，保持有效 batch 不变。当前 Whisper encoder 要求固定
3000 帧 log-Mel 输入，因此批处理明确补齐到 30 秒，并在进入特征提取前拒绝超长样本，不宣称采用
动态特征 padding 降低计算量。训练数据加载默认使用 16 个 CPU worker 和每 worker 4 个预取 batch，
服务器 CPU 不作为稀缺资源；若 GPU 利用率仍有明显空洞，再提高 worker 数而不改变实验协议。
正式训练前运行 100 optimizer steps，记录 step time、峰值显存、
音频吞吐和预计总时长：

1. 单次预计 `<=8 h`：20 h、最多 3 epoch；
2. 超过 8 h：先统一改成最多 2 epoch；
3. 仍超过 8 h：L1/L2 同时统一降成 10 h；
4. 不能只缩短某一模型。

训练循环每 10 optimizer steps写入一次结构化指标，终端每 20 步动态刷新一次。每个 epoch 结束解码
`dev_model` 并计算 CER。选择 `dev_reverb_cer` 最低且 clean 退化可接受的 checkpoint，只保存
adapter、训练状态、配置、依赖版本、base model revision 和 SHA。

### 10.6 终端与日志职责

终端只服务于训练监控和超参数调整：启动时显示模型、LoRA、数据规模、有效 batch、学习率与 GPU；
训练中显示 epoch/step、即时与 EMA loss、学习率、梯度范数、速度、显存和 ETA；每个 epoch 结束只显示
clean CER、总体 reverb CER、重混响 CER、当前最佳值和 checkpoint 状态。只有 NaN、梯度异常、显存
或数据吞吐问题才额外打印 warning。

完整信息写入 run 目录，不在终端展开：

```text
runs/<experiment_id>/
  run_config.json          冻结协议和全部超参数
  environment.json         Git、Python、CUDA 和依赖版本
  data_audit.json          manifest/RIR 哈希和泄漏检查
  train_metrics.jsonl      每 10 step 的训练指标
  eval_metrics.jsonl       每个 epoch 的完整 CER、S/D/I 和分层结果
  eval_by_rt60.json        各 epoch 的五档 RT60 CER
  predictions.jsonl        逐条参考、识别结果和条件
  warnings.jsonl           NaN、OOM、异常音频和性能告警
  training_summary.json    最佳 epoch、耗时、峰值显存和 checkpoint
```

`src/robust_asr/training/reporting.py` 是训练循环的统一输出接口。结构化日志支持同一配置续跑；若
run 目录中的配置、环境或数据审计与本次启动不一致，则拒绝覆盖，避免把两个实验混在一起。

## 11. 实验和消融矩阵

### 11.1 阶段 A：数据与管线验收

| ID | 工作 | 通过条件 |
|---|---|---|
| A0 | AISHELL 下载与官方 split 审计 | checksum、时长、speaker/utterance 一致 |
| A1 | 20 h/dev/test manifest | 时长误差、覆盖、互斥和 SHA 通过 |
| A2 | RIR bank | shape、room-disjoint、RT60 容差、几何约束通过 |
| A3 | 卷积输出 | 四通道同步、完整 tail、无削波/NaN |
| A4 | 中文 normalizer/CER | 金标准单测和手算样例通过 |

### 11.2 阶段 B：冻结 Whisper 基线

| ID | 模型 | 输入 | 目的 |
|---|---|---|---|
| B0 | W0 pretrained | clean | 普通话零样本基线 |
| B1 | W0 pretrained | raw，五档 RT60 | 混响退化曲线 |
| B2 | W0 pretrained | S-WPE-10 | 同时间跨度单通道基线 |
| B3 | W0 pretrained | S-WPE-40 | 容量匹配单通道基线 |
| B4 | W0 pretrained | M-WPE-10 | 多通道前端收益 |

先用 `dev_frontend` 验证方向和管线，再冻结 WPE 后运行 test。

### 11.3 阶段 C：LoRA 消融

| ID | 比较 | 回答的问题 |
|---|---|---|
| C1 | LP1 vs LP2 | 混响适配主要需要 encoder，还是 decoder 也需要适配？ |
| C2 | W0 vs L1 | 普通话领域适配本身带来多少收益？ |
| C3 | L1 vs L2 | 同训练预算下，RIR 数据增强是否有效？ |
| C4 | L2 clean vs L1/W0 clean | MCT 是否牺牲 clean 性能？ |

rank、dropout、学习率不做完整网格；它们只在训练失败时进入诊断，不作为简历主消融。

### 11.4 阶段 D：最终全因子矩阵

模型：

```text
W0 pretrained Whisper
W1 Clean-LoRA
W2 MCT-LoRA
```

仿真测试对每个模型运行：

```text
1,000 utterances
× 5 RT60
× 4 frontends (Raw/S10/S40/M10)
= 20,000 ASR inputs/model
= 60,000 inputs total
```

核心汇总表：

| Model | Frontend | Clean | RT60 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | Robust mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W0 | Raw | CER | CER | CER | CER | CER | CER | CER |
| W0 | S-WPE-10 | — | CER | CER | CER | CER | CER | CER |
| W0 | S-WPE-40 | — | CER | CER | CER | CER | CER | CER |
| W0 | M-WPE-10 | — | CER | CER | CER | CER | CER | CER |
| W1 | ... | ... | ... | ... | ... | ... | ... | ... |
| W2 | ... | ... | ... | ... | ... | ... | ... | ... |

### 11.5 WPE 与 LoRA 互补性

定义：

\[
\Delta_{WPE}(W)=CER(W,M\text{-}WPE)-CER(W,Raw).
\]

负值表示 WPE 改善。交互量：

\[
I=\Delta_{WPE}(W2)-\Delta_{WPE}(W0).
\]

- `I≈0`：二者近似独立；
- `I>0`：MCT 后 WPE 收益减小，可能冗余；
- `I<0`：WPE 在 MCT 后收益更大，存在互补；
- 任一 delta 区间跨零：只称“未证明有收益”，不称等价。

### 11.6 外部与可选实验

核心矩阵完成后：

1. 从 [RIRS_NOISES](https://www.openslr.org/28/) 或其他明确许可来源筛选实测 RIR；
2. 先审计其是否为同步多通道阵列 RIR；
3. 多通道可用时，只比较 `W0/W2 × Raw/M-WPE`；
4. 只有单通道时，比较 `W0/W2 × Raw/S-WPE-40`；
5. 实测 RIR 不参与训练、WPE 参数选择或 LoRA checkpoint 选择。

若时间仍充足，使用冻结 Paraformer-AISHELL 对 Raw 与最佳 WPE 做小规模跨模型推理验证；不微调、
不重复 LoRA 全矩阵。AISHELL-4、Paraformer 微调和噪声混合继续排除。

## 12. 评测与统计

### 12.1 主指标

主指标为 normalized corpus CER，并报告：

```text
reference chars
substitutions
deletions
insertions
CER
```

所有比较以 utterance 严格配对。对 CER 差值进行 10,000 次 utterance-level paired bootstrap，
seed `2026`，报告绝对百分点变化、相对改善和 95% percentile CI。

### 12.2 条件化分析

必须报告：

- RT60 五档；
- DRR 分桶；
- speaker；
- S/D/I 构成；
- leave-one-speaker-out 方向稳定性；
- 去掉影响最大 utterance 后的方向性检查。

source-array distance、房间体积、声源高度和方位角作为几何协变量保留，用于检查 DRR–CER
关系是否由特定几何分布驱动。距离分桶图是可选诊断，不作为项目主结论。

### 12.3 信号指标

模拟数据有 clean source，可在做时间/尺度对齐后计算 SI-SDRi 和 ΔSTOI；可选增加 SRMR。它们用于
回答“信号去混响是否与 CER 同向”，不能替代 CER，也不能单独决定前端或 checkpoint。

### 12.4 解码异常

解码异常诊断记录：

- hypothesis 字符数和字符速率；
- 重复 n-gram；
- token limit；
- compression ratio（若后端提供）；
- no-speech/text 冲突；
- 推理异常、空文本和超长文本。

主协议不自动回退 Raw，不依据 reference 选分支。异常策略只能用于诊断，不能覆盖真实前端结果。

## 13. 缓存、复现和防泄漏

缓存 key 至少包含：

```text
audio SHA
RIR SHA / frontend config SHA
model revision / adapter SHA
decoder config SHA
normalizer version SHA
```

配置变化必须使缓存失效。所有随机过程使用显式 seed；manifest 和配置写入原子文件；中断后允许
逐样本续跑。正式 test 运行前生成冻结清单：

- 代码 commit；
- 环境依赖版本；
- 全部配置 SHA；
- train/dev/test manifest SHA；
- RIR bank 审计 SHA；
- base model revision；
- adapter checkpoint SHA；
- normalizer 和 decoder 配置。

双重防泄漏：语音遵循 AISHELL 官方 split，RIR 同时保证 room/geometry/seed split，不允许只做
utterance 隔离而复用同一房间。

## 14. 测试计划

### 14.1 单元测试

- AISHELL ID、speaker 和 transcript 解析；
- normalizer 的繁简、标点、空格、英文和数字样例；
- CER/S/D/I 手算样例；
- RIR seed 确定性和四通道 shape；
- 房间几何、墙距和 source-array distance；
- RT60 容差与拒绝重采样；
- 完整卷积长度、共同增益和 clipping guard；
- WPE 输出 shape/长度/有限值/参考通道；
- train RIR sampler 的 50/50 分布和跨 epoch 确定性；
- LoRA 目标模块名、参数量和 base model 全冻结；
- cache key 对配置变化敏感；
- paired bootstrap 固定 seed 可复现。

### 14.2 集成测试

用 2–5 条语音完成：

```text
clean → 4ch RIR → Raw/S-WPE/M-WPE → Whisper → CER → summary
```

再用 30–50 条完成 smoke run，检查显存、训练 loss、缓存续跑、WPE 稳定性和汇总表，之后才允许
启动完整 dev/test 或 LoRA 训练。

## 15. 一周实施顺序

| 天 | 主任务 | 必须产物 |
|---|---|---|
| Day 1 | 新模块骨架、依赖、AISHELL 下载/审计、20 h/dev/test manifest | data audit、manifest SHA、测试 |
| Day 2 | Pyroomacoustics RIR bank、RT60/DRR、卷积和 WPE 封装 | RIR audit、WPE smoke、试听样例 |
| Day 3 | 冻结 Whisper clean/raw/WPE dev 基线、100-step 训练 benchmark | WPE 配置冻结、算力预算 |
| Day 4 | LP1/LP2 两个 5 h 短程 LoRA | 位置消融表、选定 target modules |
| Day 5 | 正式 Clean-LoRA 与 MCT-LoRA | 两个 adapter、训练曲线、dev CER |
| Day 6 | 60,000 输入的缓存式最终评测与 bootstrap | 主结果 CSV/JSON、CI、错误样例 |
| Day 7 | 实测 RIR 小测（若核心完成）、PNG 图、README/手册和简历材料 | 最终报告、图、复现命令、结论边界 |

若前序延期，优先级为：数据审计 > frozen baseline > 两个正式 LoRA > 核心 2×2 互补矩阵 > 完整
WPE 消融 > 实测 RIR > Paraformer。不得为了外部扩展牺牲核心 test 的完整性。

## 16. 验收和停止条件

### 16.1 项目完成条件

以下全部满足才称核心项目完成：

- AISHELL/RIR 双重 split 审计通过；
- clean/raw RT60 曲线完成；
- Raw、S-WPE-10、S-WPE-40、M-WPE-10 在 frozen Whisper 上完成；
- LoRA 位置短程消融完成；
- Clean-LoRA 与 MCT-LoRA 同预算正式训练完成；
- W0/W1/W2 的核心矩阵和 paired CI 完成；
- test 未参与调参；
- 失败样例和结论边界写入文档；
- 代码、配置、manifest、缓存和结果可复现。

### 16.2 效果判定

以下是解释标准，不是强制达到的结果：

- WPE/MCT 的 CER 改善应报告 CI；区间跨零只称未证明；
- clean CER 绝对退化超过 `0.5 pp` 时，MCT 必须标记 clean trade-off；
- Raw CER 未随 RT60 明显恶化时，先审计 RIR/幅值/映射，不启动大规模 LoRA；
- WPE 信号指标改善但 CER 变差时，保留结果并分析失真，不按信号指标替换主结论；
- MCT 无收益时，先检查训练收敛、标签、RIR采样和缓存，再条件性尝试 `r=16`；
- test 结果不理想不能回到 test 调参，只能在新 dev 实验中提出下一版方案。

## 17. 风险与回退

| 风险 | 诊断 | 回退 |
|---|---|---|
| 4070 单次训练过久 | 100-step benchmark | 3→2 epoch；仍超时则两模型同时 20→10 h |
| 显存不足 | 记录 peak VRAM/OOM | batch 2→1，accumulation 8→16 |
| 仿真 RT60 偏差 | 实测 RT60 与目标比较 | 拒绝重采样；不使用目标值直接分桶 |
| WPE 数值失败 | solver warning/NaN/短语句 | complex128、检查条件数、只在 dev 诊断 taps/delay |
| M-WPE 没有 CER 收益 | 信号指标、错误类型、RT60 分层 | 作为负结果；必要时再研究 WPE-aware LoRA |
| MCT 损害 clean | clean full test、训练比例 | 下一版在 dev 比较 70/30；不改当前 test 结论 |
| 仿真到实测不迁移 | measured-RIR test | 明确 simulation gap，不声称真实会议室鲁棒 |
| 数字/繁简造成 CER 偏差 | strict/norm CER 与样例审计 | 固定 normalizer，格式问题单独报告 |
| 范围膨胀 | 周计划和优先级检查 | 删除 Paraformer、AISHELL-4、部署和噪声扩展 |

## 18. 最终产物

### 18.1 代码与复现

- 独立 `robust_asr` 包、CLI、配置和测试；
- AISHELL/RIR/增强/训练/评测全链路 manifest；
- adapter-only checkpoint 和环境锁定信息；
- 一条 smoke 命令、一条训练命令和一条缓存式评测命令；
- 结果 CSV/JSON 与 bootstrap 明细。

### 18.2 文档图

所有图以 PNG 生成并放在 `docs/figures/robust_asr/`：

1. 系统与数据流图；
2. Raw CER–RT60 曲线；
3. S-WPE/M-WPE CER–RT60 对比；
4. W0/W1/W2 × frontend 热力图；
5. clean 与 robust CER trade-off；
6. substitution/deletion/insertion 构成；
7. 信号指标变化与 CER 变化散点图；
8. 典型成功/失败样例的波形或频谱图。

不得生成或提交 SVG。

### 18.3 项目包装

最终叙事必须回答：

```text
问题：普通话 ASR 在不同混响强度下如何退化？
数据：如何构造 room-disjoint、可控且可复现的多通道 RIR 数据？
方法：WPE 与 LoRA multi-condition training 各解决什么？
实验：单/多通道、clean/MCT、frozen/adapted 如何公平消融？
结果：哪些条件真正降低 CER，哪些信号改善没有转化成 ASR 收益？
边界：仿真结论能否迁移到未见实测 RIR，哪些问题尚未覆盖？
```

在新结果完成前，不提前写具体 CER 改善数字。历史英文项目只通过 Git 历史审计，不重新并入活动代码树。

## 19. 参考实现与资料

- [AISHELL-1 / OpenSLR 33](https://www.openslr.org/33/)
- [Pyroomacoustics room simulation](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html)
- [NARA-WPE](https://github.com/fgnt/nara_wpe)
- [OpenAI Whisper paper](https://cdn.openai.com/papers/whisper.pdf)
- [Whisper-small multilingual model](https://huggingface.co/openai/whisper-small)
- [Hugging Face PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora)
- [LoRA-Whisper](https://arxiv.org/abs/2406.06619)
- [RIRS_NOISES / OpenSLR 28](https://www.openslr.org/28/)
