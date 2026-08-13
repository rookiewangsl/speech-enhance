# ASR 后端与识别评测方案

最后更新：2026-08-14

本文记录项目 ASR 后端接入和识别评测协议。当前状态是**方案冻结、核心管线已实现，
官方 VCTK reference 已完成结构审计，development balanced 子集的正式 CPU 四路运行已完成**。
人工抽查和 development/validation/official test 全量运行仍待完成，因此子集数字只作实现验收
和问题定位，不作为主实验结论。现有增强结果见
[06_结果与结论说明.md](06_结果与结论说明.md)，数据边界见
[01_数据集说明.md](01_数据集说明.md)。

## 1. 研究问题与范围

主问题固定为：

> 在不微调 ASR、保持模型权重和解码配置完全一致的前提下，语音增强是否改善固定 ASR
> 后端的英语识别准确率？

这个问题评价的是增强前端对下游识别的实际影响，而不是选择最强 ASR、微调 Whisper，或用
ASR loss 重新训练增强器。SI-SDR、STOI 和听感变好都不等价于 WER 必然下降，因此 ASR
结果作为独立证据报告。

当前阶段不包括：

- Whisper 微调、prompt tuning 或 ASR-aware enhancement；
- 多个 Whisper 尺寸之间的模型选择；
- 流式 Whisper 或端点检测系统；
- 用 clean 音频的 ASR 输出代替人工 reference；
- 根据 validation 或 official test 结果反向调整解码参数。

DNSMOS P.835 是后续并行扩展，不阻塞 ASR 主线。真实自由说话的麦克风录音没有人工文本时
不报告 WER；只有朗读固定 prompt 或完成独立人工转写后才计算 WER。

## 2. 已冻结决策

| 项目 | 冻结选择 |
|---|---|
| ASR 实现 | 官方 `openai-whisper` 本地实现 |
| 模型 | `small.en`，唯一主后端，不微调 |
| 权重 | 官方权重，SHA-256 `f953ad0f...e0872` |
| 软件版本 | `openai-whisper==20250625`、`torch==2.13.0`、Python 3.12.10 |
| 参考设备 | CPU + FP32 |
| 加速设备 | 本机 MPS 一致性检查失败；v1 正式运行固定为 CPU |
| 任务/语言 | `task=transcribe`、`language=en` |
| 解码器 | beam search，`beam_size=5`、`patience=1.0` |
| 随机解码 | `temperature=0.0` 单一温度，不使用 temperature fallback |
| 上下文 | 无 initial prompt，`condition_on_previous_text=false` |
| 内部阈值 | 使用所固定 `openai-whisper` 版本的默认值，但把实际值写入实验配置 |
| 主指标 | 标准化后的 corpus WER |
| 附加指标 | substitution、deletion、insertion、ASR-only RTF、端到端 RTF |
| 输入条件 | clean、noisy、MCRA + DD-Wiener、RNNoise R3 |
| 主结果 | 8-speaker validation |
| 诊断 | development，重点检查 high-SNR 和 babble |
| 补充结果 | official test；有历史开发接触，不称为完全盲测 |

精确版本、阈值和完整权重 SHA-256 已写入
[`configs/asr_whisper_small_en.json`](../../configs/asr_whisper_small_en.json)。本机独立环境实测为
Python 3.12.10、`openai-whisper` 20250625、PyTorch 2.13.0、NumPy 2.5.2 和 FFmpeg 8.1.2；
模型权重 SHA-256 为
`f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872`。若任一冻结值改变，
必须产生新的实验版本和输出目录。

## 3. 总体数据流

```text
官方 VCTK transcript
        │ 以 speaker_id + utterance_id 严格映射并审计
        ▼
ASR reference manifest ──────────────────────────────────────────────┐
                                                                    │
VoiceBank clean ───────────────────────────────→ clean ──────────────┤
VoiceBank noisy ───────────────────────────────→ noisy ──────────────┤
             ├→ 冻结 MCRA + DD-Wiener ─────────→ classical ──────────┤
             └→ 冻结 RNNoise R3 + 离线对齐补偿 → rnnoise_r3 ─────────┤
                                                                    ▼
                                固定 Whisper small.en beam search
                                                                    │
                         raw/normalized hypothesis + timing
                                                                    ▼
                     WER / S / D / I / RTF / 分层统计
```

四路输入按同一个 utterance `id` 配对。任何聚合都必须建立在相同 utterance 集合上，不能
静默丢弃失败样本后比较均值。

## 4. 实现路线

### 4.1 阶段一：参考文本获取与对应关系审计

优先下载官方 VCTK 0.92 完整包，只把其中的人工 transcript 作为 VoiceBank reference 来源。
由于 VoiceBank+DEMAND 制作时引用的是 VCTK 源语音，正式采用 0.92 文本前还要把它与
VoiceBank 官方页面记录的来源版本和实际 utterance ID 覆盖情况交叉核对；如果发现版本差异，
应改用 VoiceBank 明确引用的官方历史版本，不能用第三方镜像静默补齐。原始包和解压内容
保留在外置盘，不进入 Git。下载记录至少包含来源 URL、数据版本、许可信息、文件大小和
checksum。

映射键使用现有 manifest 的 `id`，例如：

```text
VoiceBank: p232_001.wav
speaker:   p232
utterance: p232_001
VCTK:      txt/p232/p232_001.txt
```

不能只因文件 stem 相同就默认版本完全对应。生成 reference manifest 前必须完成：

1. development、validation、official test 中每个 `id` 恰好匹配一份 transcript；
2. speaker 目录和文件名前缀一致；
3. 无缺失、空文本、重复 ID 或未消费的目标 ID；
4. 分层抽样人工核对 clean 音频内容与 transcript，覆盖不同 speaker 和 partition；
5. reference manifest 保存源 transcript 原文和源文件相对路径；正式转写时再与 hypothesis
   一起调用同一个冻结 Whisper normalizer；
6. 输出覆盖率、重复项、缺失项和抽样核对记录的审计报告。

实现生成独立的 ASR reference manifest，不改写现有 VoiceBank manifest。字段为：

```text
id, speaker_id, split, reference_raw,
transcript_source, transcript_version, transcript_relative_path
```

这里有意不在准备脚本中预先生成 `reference_normalized`：准备阶段不依赖 ASR 环境，避免使用
未冻结或不一致的 normalizer。`evaluate_asr.py` 在同一个进程中用同一个
`EnglishTextNormalizer` 同时生成 `reference_normalized` 和 `hypothesis_normalized`；评分器会
拒绝一侧规范化、另一侧未规范化的输入。

VCTK 文本及派生 reference 是否进入 Git，必须先按其许可完成 attribution 和再分发检查；
在此之前只提交下载配置、审计代码和不包含正文的统计报告。

### 4.2 阶段二：独立 ASR 环境与模型冻结

ASR 使用 `.venv-asr` 独立 Python 环境，避免给当前轻量 DSP 环境强制加入 PyTorch。环境记录
包括：

- Python、`openai-whisper`、PyTorch、NumPy、WER 评分库和 FFmpeg 版本；
- 操作系统、CPU、MPS 可用性和精度；
- `small.en` 权重文件大小与 SHA-256；
- 完整解码配置和内部阈值；
- 代码 Git revision 和配置摘要哈希。

CPU + FP32 是参考路径。2026-08-13 的本机 probe 在同一 utterance 的 clean、noisy、
MCRA + DD-Wiener 和 RNNoise R3 四路中发现设备差异。最终严格身份对照报告状态为 `failed`
（0/4 normalized exact）：CPU 四路均为 `please call stela`，MPS 四路均为
`please call stella`。MPS 聚合 ASR RTF 为 0.423，快于 CPU 的 0.655，但速度不能抵消识别
结果变化。早期同代码 MPS 强制复跑还曾有 3/4 normalized hypothesis 改变，而对应的 CPU
强制复跑保持 4/4 一致；因此当前协议明确回退 CPU，MPS 不得用于正式全量结果。该 probe 很小，
不足以评价一般 MPS 准确率，却已经足以拒绝其作为 v1 的可替代执行设备。设备验收以 normalized
hypothesis 完全一致为最低条件；只要存在无法
解释的识别差异、非有限值、算子回退或运行不稳定，正式全量结果就使用 CPU + FP32。四路
输入必须使用同一设备、精度、模型进程和解码配置。

### 4.3 阶段三：四路音频统一生成与缓存

四种输入定义为：

| condition | 音频定义 |
|---|---|
| `clean` | manifest 中原始 clean 16 kHz mono WAV，不经过增强 |
| `noisy` | manifest 中原始 noisy 16 kHz mono WAV，不经过增强 |
| `mcra_dd_wiener` | 冻结 MCRA + DD-Wiener，`alpha_dd=0.92`、`gain_floor=0.20` |
| `rnnoise_r3` | official pretrained RNNoise R3，16 kHz streaming wrapper，离线固定延迟补偿 |

所有输入统一检查 16 kHz、mono、finite、样本数和 utterance ID。clean/noisy 不重新做 peak
normalization；增强输出不得为了 ASR 单独调增益。每个缓存文件同时记录源音频摘要、方法配置
摘要、Git revision、相关实现源码 SHA-256 和输出摘要，避免代码或配置改变后误用旧音频。

中间 WAV 保存在外置盘，不进入 Git。缓存键至少包含：

```text
(utterance_id, condition, source_audio_sha256, source_manifest_digest,
 enhancement_config_digest, output_audio_sha256)
```

### 4.4 阶段四：固定 Whisper 批量转写

每条音频以完整 utterance 离线转写，不额外切成流式块。冻结解码配置为：

```text
model                    = small.en
task                     = transcribe
language                 = en
temperature              = 0.0 only
temperature_fallback     = disabled
beam_size                = 5
patience                 = 1.0
best_of                  = None
length_penalty           = None
initial_prompt           = None
condition_on_previous_text = False
```

运行时保存 raw hypothesis、segment metadata、实际 temperature、平均 log probability、
compression ratio、no-speech probability、处理耗时和错误信息。正式运行前用固定音频 warm-up；
ASR-only 计时包括 log-Mel 特征、模型推理和文本解码，但模型加载、首次编译、音频读取和
文件写出不计入 ASR-only RTF。

任务必须支持原子写入和中断续跑。转写缓存键至少包含：

```text
(utterance_id, condition, audio_sha256, asr_config_digest, model_sha256,
 evaluator_code_sha256, runtime_identity_digest)
```

只有完整匹配缓存键的结果才能跳过。异常退出、空文件、字段不完整或配置摘要不同的结果必须
重新运行。`runtime_identity_digest` 覆盖 device、精度、Python/Torch/NumPy 版本、机器和线程
数，防止 CPU/MPS 或不同线程的结果与 RTF 混用。命中 ASR-only 缓存时仍会用当前输入 manifest
重新组合前端 provenance 和 end-to-end RTF，避免相同音频但更新后的增强 metadata/耗时被旧
结果覆盖。空 hypothesis 若是 Whisper 的合法输出则保留并计入 deletion，不能当作失败样本
删除。

### 4.5 阶段五：文本规范化与 WER 评分

reference 和 hypothesis 使用固定 `openai-whisper` 版本提供的 English text normalizer 做
同一套规范化，并同时保存原文和规范化文本。不得只规范化其中一侧，也不得看到结果后修改
规则。汇总 CLI 默认强制整个 corpus 使用 normalized/normalized；raw/raw 只可通过显式
`--allow-raw` 作为诊断模式，不能作为主结果，并且汇总 JSON 会记录实际 `text_mode`。

主指标使用 corpus WER：

\[
\mathrm{WER}=\frac{S+D+I}{N},
\]

其中 `N` 是整个统计集合的 reference word 总数。主结果不使用逐句 WER 的简单算术平均，
避免短句得到过大权重。同时报告：

- substitution、deletion、insertion 的总数及除以 `N` 的比例；
- 相对 noisy 的绝对 WER 变化和相对 WER reduction；
- 每条 utterance 的 reference/hypothesis、S/D/I/N 和 WER；
- 方法间基于同一 utterance 的 paired bootstrap 区间；
- clean 作为 ASR 上限参考，不称为增强方法。

### 4.6 阶段六：RTF 定义

分别报告两种 RTF：

```text
ASR-only RTF = synchronized ASR backend time / audio duration
end-to-end RTF = (enhancement processing time + ASR backend time) / audio duration
```

其中 ASR backend time 包含 log-Mel 特征、模型推理和文本解码，不包含模型加载、音频读取和
结果写盘。clean/noisy 的 end-to-end 前端时间按零处理；MCRA + DD-Wiener 和 RNNoise 使用
各自实测处理时间。设备、精度、线程数、warm-up 和计时边界必须随结果记录。MPS 计时必须在
开始和结束处同步设备，避免把异步提交时间误当作完整推理耗时。

### 4.7 阶段七：分阶段运行和分层分析

执行顺序固定为：

1. `development_balanced_5`：环境、映射、缓存、评分和四路完整冒烟；
2. development full：条件诊断，重点检查 `15 dB` 和 `babble`；
3. validation full：8 个未见 speaker 的主结果；
4. official test：补充结果，重点检查 `17.5 dB`，但不称为完全盲测。

主汇总按 `condition` 报告，并进一步按以下字段分层：

- noise type；
- input SNR；
- speaker；
- partition；
- reference 长度，可作为辅助诊断而非核心结论。

重点验证的假设是：增强可能在低 SNR 降低 insertion 或总体 WER，却在 high-SNR 和 babble
条件因过处理增加 deletion/substitution。该方向只是预注册诊断假设，最终结论必须由实际
数据决定。

任何在 development 上发现的问题都只能修正实现缺陷，或产生显式的新协议版本。查看
validation 后不得继续微调 ASR/增强参数并覆盖同名结果。

## 5. 已实现代码与产物

当前入口保持职责分离：

```text
configs/asr_whisper_small_en.json       # 模型、解码、规范化和计时协议
scripts/prepare_asr_references.py       # transcript 映射与审计
scripts/export_asr_inputs.py            # 四路输入生成、校验与缓存
scripts/evaluate_asr.py                 # 固定 Whisper 转写和续跑
scripts/compare_asr_devices.py          # CPU/MPS 严格配对一致性报告
scripts/summarize_asr_metrics.py        # WER/S/D/I/RTF 聚合与分层
tests/test_asr_references.py
tests/test_asr_inputs.py
tests/test_evaluate_asr.py
tests/test_compare_asr_devices.py
tests/test_asr_scoring.py
```

大体积文件位于外置盘：

```text
VCTK 原始包和解压内容
Whisper small.en 权重
四路中间 WAV
可重建的模型缓存
```

仓库只保留代码、冻结配置、许可/来源记录、审计摘要、逐条评分 CSV/JSON、汇总 JSON/CSV 和
PNG 图。按照仓库规则，新增文档图只使用 PNG 并放在 `docs/figures/`，不添加 SVG。

建议结果结构为：

```text
outputs/asr/<protocol_version>/
  environment.json
  reference_audit.json
  device_equivalence.json
  hypotheses/<split>/<condition>.jsonl
  metrics/<split>/per_utterance.csv
  metrics/<split>/summary.json
  metrics/<split>/by_noise.csv
  metrics/<split>/by_snr.csv
  metrics/<split>/by_speaker.csv
```

## 6. 自动检查与验收条件

正式发布 ASR 结果前必须同时满足：

- 三个 partition 的目标 utterance transcript 覆盖率为 100%，无重复和空 reference；
- 四种 condition 对每个目标 utterance 均存在且通过音频契约检查；
- Whisper 模型文件、环境版本、解码参数和配置摘要完整记录；
- 所有实际 segment temperature 均符合冻结的单温度协议；
- 中断恢复不会重复覆盖有效结果，也不会复用配置不匹配的缓存；
- WER/S/D/I 用人工构造样例验证，且满足 `WER=(S+D+I)/N`；
- 聚合使用完全相同的 utterance 集合，失败或空 hypothesis 不被静默删除；
- validation 只使用冻结配置，official test 明确标注历史接触边界；
- 汇总至少包含 overall、noise、SNR 和 speaker 四个层级；
- ASR-only 与 end-to-end RTF 的计时边界、设备和精度可追溯；
- 随机抽查 reference、四路音频、hypothesis 和 edit alignment；
- 大体积 WAV、VCTK 原始数据和模型权重不进入 Git。

截至 2026-08-14，仓库全套 172 项测试通过，Python 静态编译和 `git diff --check` 通过。
VCTK 0.92 官方完整包已通过精确字节数、全 ZIP CRC、官方 MD5 和本地 SHA-256 校验。
三个 partition 共 12,396 条目标 utterance 的 reference 结构覆盖率为 100%：
development 8,348、validation 3,224、official test 824，缺失、重复、空文本和 speaker 不匹配均为零。
审计报告仍把每个 partition 的 3 条人工听辨抽查标为 `pending`；结构检查通过不等于这一
人工步骤已完成。

### 6.1 Development balanced 正式运行

`development_balanced_5` 使用 20 个 speaker、每人 5 条，共 100 个 utterance；四路完全配对，
reference 共 744 词。固定 CPU + FP32 的结果为：

| condition | S / D / I | corpus WER | 相对 noisy 的绝对变化 | paired bootstrap 95% CI | ASR / E2E RTF |
|---|---:|---:|---:|---:|---:|
| clean | 14 / 0 / 3 | 2.28% | -3.36 pp | [-5.92, -1.19] pp | 0.572 / 0.572 |
| noisy | 30 / 3 / 9 | 5.65% | 0.00 pp | [0.00, 0.00] pp | 0.577 / 0.577 |
| MCRA + DD-Wiener | 29 / 3 / 9 | 5.51% | -0.13 pp | [-1.08, 0.93] pp | 0.568 / 0.576 |
| RNNoise R3 | 73 / 20 / 212 | 40.99% | +35.35 pp | [5.16, 91.43] pp | 0.653 / 0.668 |

这个子集上，MCRA + DD-Wiener 只比 noisy 少 1 个词错，区间跨零，尚无证据说明它稳定
改善识别。RNNoise 结果中的 insertion 激增高度集中于 `p244_166 / babble / 5 dB`：该
3.496 s 音频生成 195 词重复串，其中 185 个 insertion，compression ratio 为 17.88，ASR RTF
为 6.196。四路样本数、时长、文件摘要和 RNNoise 延迟补偿均通过校验，相邻样本运行时间
也恢复正常，因此它作为真实的固定后端失败保留，不从主统计剔除。
长尾诊断与此一致：RNNoise 的逐句 ASR RTF median / p95 / max 为
0.605 / 0.786 / 6.196，100 条中只有该 1 条的 segment compression ratio 超过 2.4；其他三路
的超阈值计数都为零。

![p244_166 clean、noisy 与两路增强的共标尺幅度频谱](../figures/asr_p244_166_spectrograms.png)

上图把 clean reference 置于第一行，四幅图使用同一 `-100–-20 dBFS` colorbar，不对每幅图
单独自动拉伸。RNNoise 图中的整体
能量明显降低，主要语音区间出现不均匀断裂，高频辅音线索也比 noisy 和 MCRA 更弱。
这与人工试听中“目标人声被清除、后半段模糊”的感受一致。
数值上，noisy 与 clean 的波形相关系数为 0.845，RNNoise 降至 0.465；相对 clean 的
SI-SDR 从 noisy 的 3.98 dB 降至 -5.60 dB。在 0.75–1.00 s、1.75–2.00 s 和
2.25–2.50 s，RNNoise 比 noisy 分别低 17.3、15.9 和 15.3 dB，说明这不是等比例
调小音量，而是随时间变化的过抑制。将 noisy 整体衰减 8.03 dB 后 Whisper 仍正确识别；
将 RNNoise 整体放大 8.03 dB 后仍生成高压缩比重复串，进一步排除“单纯音量过小”。

#### 6.1.1 为什么会形成大量 insertion

这条 RNNoise 输出让 Whisper encoder 得到模糊、断裂的声学证据，但
`no_speech_prob=0.552` 仍低于 0.6，因此解码不会跳过。decoder 是自回归语言模型：当
beam search 进入“`this is not a club ...`”这个局部高概率循环后，后续 token 主要受它自己
已生成的前缀驱动，而受残缺声学证据的约束较弱。当前协议没有 repetition penalty，
`condition_on_previous_text=false` 也只防止跨 segment 携带历史，不阻止当前 segment 内部重复。
最终文本含 223 个 tokenizer token，几乎耗尽默认 224-token `sample_len`，并在短语中途截断；
beam size 5 还使这个长序列的计算量明显增大。编辑距离在 10 个 reference word 上对齐 195 个
hypothesis word，所以额外重复词被记为 185 个 insertion。

#### 6.1.2 是低 SNR 还是 babble

balanced 子集的配对诊断表明两者都有影响，但 **babble 是更稳定的失败轴，低 SNR 进一步
放大错误，两者交互时才出现本次灾难性 insertion**：

| 分组 | utterances | noisy WER | RNNoise WER | RNNoise 更差 / 相同 / 更好 | 平均 ΔSI-SDR |
|---|---:|---:|---:|---:|---:|
| babble，0–5 dB | 4 | 20.00% | 753.33% | 4 / 0 / 0 | -6.94 dB |
| babble，10–15 dB | 5 | 0.00% | 40.00% | 3 / 2 / 0 | -12.69 dB |
| 非 babble，0–5 dB | 43 | 8.75% | 16.25% | 13 / 25 / 5 | +7.25 dB |
| 非 babble，10–15 dB | 48 | 2.23% | 3.62% | 7 / 38 / 3 | +1.53 dB |

babble 共 9 条，RNNoise 的 SI-SDR 在 9/9 上都下降，平均下降 10.13 dB，WER 在 7/9
上变差且没有一条变好。其中 0–5 dB 的 4 条 babble 产生 197 个 insertion，10–15 dB 的
5 条没有 insertion，却仍有 40.00% WER：这说明 babble 过抑制并不只存在于低 SNR，
但低 SNR 更容易把模糊声学输入推入解码长尾。作为独立证据，非 babble 的 0–5 dB 也从
8.75% 上升到 16.25%，因此低 SNR 本身同样有贡献。

机制上，RNNoise 估计的是单通道频带增益和 VAD，没有目标 speaker 条件，也不是多说话人
源分离模型。babble 干扰本身同样具有语音谱包络、基音和谐波结构，使“保留哪个人声”变成
它未被设计来解决的问题。balanced 子集还显示 meeting 是第二个需要追踪的语音型噪声：
8 条中 RNNoise/noisy WER 为 30.36%/1.79%，但样本过少，只作预警。

以上分层是基于 100 条 speaker-balanced 子集的探索性诊断，并非 noise/SNR-balanced 推断。
即使只做敏感性分析并排除 `p244_166`，RNNoise/noisy WER 仍为 15.12%/5.72%。
是否为系统性问题必须在 development full 上用相同预注册分层复验。

这个长尾与冻结解码协议直接相关：在当前 `openai-whisper` 实现中，
`compression_ratio_threshold=2.4` 用于触发 temperature fallback，不是事后删除输出的过滤器。
v1 只有单一 `temperature=0.0`，所以唯一尝试超阈值后仍会被返回。为保持预注册配置，不在
看到结果后为单条样本开启 fallback。后续可把“启用多温度 fallback 或显式重复抑制”作为
新协议的 ASR 稳健性消融，但不覆盖 v1 结果。

转写结果已重跑验证缓存：`cache_hits=400`、`transcribed_rows=0`，未发生重复推理。
完整 summary/per-utterance JSONL 和大体积音频继续保留在外置盘。下一阶段仍按冻结顺序运行
development full，再运行 validation full；在后者完成前不回答主研究问题。

## 7. 预期解释边界

实验可能得到总体改善、总体退化或只在部分条件改善，三种结果都有效。最终只允许做以下
类型的结论：

- “在固定 Whisper `small.en` 和冻结解码协议下，某增强方法相对 noisy 的 WER 如何变化”；
- “变化主要来自 substitution、deletion 或 insertion 中的哪一项”；
- “结论是否集中在特定 noise、SNR 或 speaker 条件”。

不能把一个固定 ASR 后端上的结果外推为所有 ASR 系统的普遍结论，也不能用 WER 单独替代
感知质量、实时性和人工试听。ASR、SI-SDR/STOI、DNSMOS P.835、RTF 和听感共同构成完整
评价链。
