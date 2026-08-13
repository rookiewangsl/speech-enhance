# 实验结果与 ASR 分析

最后更新：2026-08-14

## 1. 先区分两种完成度

| 任务 | 当前规模 | 状态 | 能否作为最终结论 |
|---|---:|---|---|
| 增强 validation | 3,224 对、8 个未见 speaker | 完成 | 是，当前主要增强证据 |
| 增强 official test | 824 对、2 个 speaker | 完成 | 补充；历史接触过 |
| ASR balanced | 100 utterances × 4 路 = 400 输入 | 完成 | 只作管线验收和问题定位 |
| ASR development full | 8,348 × 4 = 33,392 输入 | 未运行 | 否 |
| ASR validation full | 3,224 × 4 = 12,896 输入 | 未运行 | 否 |
| ASR official test | 824 × 4 = 3,296 输入 | 未运行 | 否 |

因此“RNNoise 的增强全量结果”已经成立，但“RNNoise 对固定 ASR 的最终影响”尚未完成。

## 2. 增强主结论

![MCRA+DD-Wiener 与 RNNoise R3 性能对比](../figures/baseline_vs_r3_performance.png)

### 2.1 未见 speaker validation

| 方法 | Mean/median SI-SDRi | Positive SI-SDRi | Mean ΔSTOI | Nonnegative ΔSTOI | Mean RTF |
|---|---:|---:|---:|---:|---:|
| RNNoise R3 | `+3.123/+3.567 dB` | `77.30%` | `-0.00312` | `58.87%` | `0.01470` |
| MCRA + DD-Wiener | `+2.093/+1.902 dB` | `96.77%` | `-0.00579` | `24.19%` | `0.00711` |

RNNoise 的 mean SI-SDRi 95% bootstrap CI 为 `[2.925,3.314] dB`。它平均抑制更强；
DD-Wiener 平均提升较小，但更多文件保持正 SI-SDRi。两者 mean ΔSTOI 都为负，不能宣称
整体提高可懂度。

### 2.2 official test 补充结果

| 方法 | Mean/median SI-SDRi | Positive SI-SDRi | Mean ΔSTOI | Mean RTF |
|---|---:|---:|---:|---:|
| RNNoise R3 | `+3.672/+3.652 dB` | `75.24%` | `-0.03282` | `0.01520` |
| MCRA + DD-Wiener | `+2.230/+2.000 dB` | `95.87%` | `-0.00684` | `0.00720` |

这些数字不包装成完全盲测 benchmark，因为早期使用过 official test 样本试听和选路线。

### 2.3 条件化结果

Validation 的 SNR 分层：

| SNR | RNNoise SI-SDRi | RNNoise ΔSTOI | DD-Wiener SI-SDRi |
|---:|---:|---:|---:|
| 0 dB | `+6.743` | `+0.0226` | `+2.156` |
| 5 dB | `+4.877` | `+0.00145` | `+2.426` |
| 10 dB | `+2.126` | `-0.0132` | `+2.176` |
| 15 dB | `-1.253` | `-0.0233` | `+1.616` |

Babble 是明确失败条件：RNNoise mean SI-SDRi `-6.718 dB`，DD-Wiener 为 `+0.778 dB`。
结论不是“R3 全场景最优”，而是：

- R3 适合低 SNR、非语音型强噪声和默认听感 Demo；
- DD-Wiener 是 high-SNR/babble 的保守 baseline；
- 真实系统不能用数据集的真值 noise/SNR 标签做旁路；
- R4 broadband noisy mixing 会重新引入结构噪声，保持关闭；
- C1 不能恢复已删除的音素，只保留为负消融。

## 3. ASR 研究问题与冻结 v1

主问题是：

> 在不微调 ASR、模型权重和解码配置完全一致时，语音增强是否改善固定 Whisper
> `small.en` 的标准化 corpus WER？

四路输入按 utterance 严格配对：`clean/noisy/mcra_dd_wiener/rnnoise_r3`。reference 来自
人工 VCTK transcript，不使用 clean ASR 伪标签。v1 固定为：

```text
openai-whisper == 20250625
small.en，权重 SHA-256 固定
CPU + FP32
language=en，task=transcribe
temperature=0.0 only
beam_size=5，patience=1.0
condition_on_previous_text=false
compression/logprob/no-speech 阈值保留，但不做 temperature fallback
```

本机 MPS probe 与 CPU 的 normalized hypothesis 不一致，因此 v1 正式运行只接受 CPU。

## 4. balanced 100 条 ASR 结果

该子集为 20 个 development speaker、每人 5 条，共 744 个 reference words：

| Condition | S / D / I | WER | 相对 noisy | paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| clean | `14 / 0 / 3` | `2.28%` | `-3.36 pp` | `[-5.92,-1.19] pp` |
| noisy | `30 / 3 / 9` | `5.65%` | `0.00 pp` | `[0,0] pp` |
| MCRA + DD-Wiener | `29 / 3 / 9` | `5.51%` | `-0.13 pp` | `[-1.08,0.93] pp` |
| RNNoise R3 v1 | `73 / 20 / 212` | `40.99%` | `+35.35 pp` | `[5.16,91.43] pp` |

这些数值支持：

1. MCRA 只少一个词错误，区间跨零；当前只能说与 noisy 基本持平。
2. RNNoise 明显损害该固定后端，但 40.99% 被一个极端样本放大。
3. 排除 `p244_166` 后，RNNoise/noisy 仍为 `15.12%/5.72%`，所以不是单个离群值造成的全部
   退化。
4. 感知指标、听感和 WER 衡量不同目标，必须并行报告。

## 5. `p244_166`：过抑制如何变成重复解码

![p244_166 四路共标尺幅度频谱](../figures/asr_p244_166_spectrograms.png)

该样本为 `babble / 5 dB / 3.496 s`。RNNoise 时频图中目标语音的辅音、共振峰过渡和后半段
连续性明显被削弱，与人工试听“人声快被削没、后半段模糊”一致。

固定 beam search 输出 195 词，其中 185 个 insertion；compression ratio 为 `17.88`，
ASR RTF 为 `6.196`。音频长度、16 kHz、文件 SHA、延迟补偿和缓存均正常。

机制是：残缺声学证据不足以持续约束自回归 decoder；beam 一旦进入局部高概率重复前缀，
后续 token 更受已生成文本驱动。`condition_on_previous_text=false` 只能阻止跨 segment 历史，
不能阻止单个 segment 内循环。最终接近 224-token sample limit，形成大量 insertion 和计算长尾。

balanced 分层表明 babble 和低 SNR 都有影响：babble 9 条中 7 条 RNNoise WER 变差、没有
一条变好；非 babble 的 0–5 dB 也从 noisy `8.75%` 上升到 RNNoise `16.25%`。两者交互时
最容易出现灾难性重复，但系统性结论仍需 full 复验。

## 6. v2 灾难保护

v2 不覆盖 v1，也不读取人工 reference 做运行时决策：

```text
v1 首次结果
  ├─ 正常 → 接受
  └─ 异常 → temperature 0.2 → 0.4 → 0.6，固定 seed、best_of=5
               ├─ 重试通过且与 paired noisy 假设词距离 ≤0.4 → 接受
               ├─ 否则 enhanced condition 回退 noisy
               └─ 无安全结果 → abstain
```

检测信号包括 compression ratio、平均 log probability、重复 4-gram、word rate、token limit
和 no-speech/text 冲突。timestamp 检查会把 Whisper 的粗粒度短音频 segment end 大量误判，
因此冻结配置关闭该项。

| 路线 | RNNoise S / D / I | WER | 相对 noisy |
|---|---:|---:|---:|
| v1 固定解码 | `73 / 20 / 212` | `40.99%` | `+35.35 pp` |
| 只取首个形状正常的 temperature retry | `67 / 26 / 27` | `16.13%` | `+10.48 pp` |
| v2 完整策略 | `62 / 16 / 27` | `14.11%` | `+8.47 pp` |
| paired noisy | `30 / 3 / 9` | `5.65%` | `0 pp` |
| condition/noisy reference oracle | `24 / 6 / 4` | `4.57%` | `-1.08 pp` |

v2 修复了 `p231_046` 和 `p244_166` 两个触发样本，均回退 noisy。它相对 RNNoise v1 减少
65.57% 错误，但 `v2-noisy` 95% CI 仍为 `[3.47,14.35] pp`，不能称优于 noisy。

这里的 oracle 不是 clean ASR。它使用人工 reference，逐 utterance 事后选择 RNNoise/noisy
两路中词错误更少者，因此不可部署。其 `4.57%` 说明 RNNoise 在部分样本上有条件价值，缺失
的是可靠的 reference-free router。

以“condition 错误数大于 paired noisy”为 harmful 标签，RNNoise 共有 27 条 harmful；当前
detector 命中 2、漏掉 25，precision `100%`、recall `7.41%`。它能抓明显解码崩溃，不能抓
所有形态正常但语义错误的增强损伤。

## 7. 工业 ASR 是否加降噪前端

会，但通常是可配置的 capture front end，而不是无条件串接一个固定 RNNoise。

官方 Whisper 没有时域降噪器：音频被重采样到 16 kHz 后直接计算 log-Mel 输入 encoder；
其默认 fallback 是解码稳健机制，不是 speech enhancement。Whisper 依靠大规模多样化训练
获得噪声鲁棒性，因此增强器可能删除模型原本能够利用的线索。
[Whisper 论文](https://cdn.openai.com/papers/whisper.pdf)、
[官方音频实现](https://github.com/openai/whisper/blob/main/whisper/audio.py)

工业系统常见 AEC、beamforming、dereverberation、AGC、noise suppression 和 VAD。例如
Microsoft Speech SDK 提供可逐项开关的音频处理栈，并明确指出面向机器识别时增强失真可能
损害准确率；NVIDIA Riva 允许用神经 VAD 过滤噪声、减少 spurious words。
[Microsoft Audio Stack](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/audio-processing-speech-sdk)、
[NVIDIA Riva ASR pipeline](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/public/asr/asr-pipeline-configuration.html)

适合前端的条件：

- 有可靠物理辅助信息：AEC reference、麦克风阵列几何、明确方向或端点；
- raw ASR 已因回声、远场、稳定机械噪声或长静音显著失败；
- 前端按下游 WER 联合训练、调强度或动态路由；
- 人工监听和 ASR 采用双路，保留 raw/noisy 安全路径。

不适合无条件增强的条件：

- high-SNR、近讲、raw ASR 已经很好；
- 单通道 babble、重叠说话人和目标身份不明确；
- 极低 SNR 下激进非线性抑制；
- 只验证 SI-SDR/STOI/听感，没有 paired WER；
- 输入已经经过未知厂商的 AEC/NS/AGC，再重复处理。

研究证据表明，单通道非线性增强产生的人工伪影可能比残留自然噪声更损害 ASR；联合训练、
ASR loss、动态增强强度或混回部分 observation 能缓解失配。
[处理伪影与 ASR](https://arxiv.org/abs/2404.14860)、
[Google SNRi target training](https://research.google/pubs/snri-target-training-for-joint-speech-enhancement-and-recognition/)

当前项目的部署判断是：

```text
ASR 默认：noisy/raw
人工监听和默认降噪 Demo：RNNoise R3
保守 ASR 前端候选：MCRA + DD-Wiener
RNNoise→ASR：只在 full WER 验证或可靠 router 允许时启用
```

## 8. 全量 ASR 计划

### 阶段一：development full

运行 `8,348×4=33,392` 路 v1，再复用 v1 运行 v2。目标是稳定估计总体、noise、SNR、speaker、
错误类型、重复长尾、detector precision/recall 和 oracle 空间。

v1 永不修改。若根据 development full 改进 router，必须创建 `v2.1`，不能覆盖 v2。

### 阶段二：validation full

在 8 个未见 speaker 的 `3,224×4=12,896` 路上一次性确认。进入前冻结模型、配置、normalizer、
阈值、router 和汇总规则；validation 上不再调参。这一 split 是最终 ASR 泛化主证据。

### 阶段三：official test

运行 `824×4=3,296` 路作为补充。必须披露历史接触边界。

按 balanced 平均时长和 CPU RTF 粗估：development v1 约 15–17 小时，validation 约 6–7 小时，
official test 约 1.5–2 小时。v2 复用 v1，只对触发样本新增推理。

## 9. partition 如何比较

当前没有训练 Whisper/RNNoise，因此 development 不是“ASR 训练集”，而是诊断和阈值开发集。

主要比较必须在同一 split 内进行：

```text
ΔWER(condition) = WER(condition) - WER(paired noisy)
```

每个 split 分别报告 `WER/S/D/I/ΔWER/paired bootstrap CI/RTF`，并按 noise/SNR/speaker 分层。
跨 split 应比较 `condition-noisy` 的效果是否一致，而不是把 development RNNoise 与 validation
noisy 直接相减。

三个 split 的 micro-average 可以作为附加统计，但不能代替分 split 结果，因为 8,348 条
development 会支配总数。

如果未来训练 learned router、ASR-aware enhancement 或微调 Whisper，必须从 development
内部再划分 train/dev，保持 validation 和 official test reference 完全不进入训练或阈值选择。

## 10. 允许下的当前结论

可以说：

- RNNoise 在增强 validation 上平均抑制更强，但 high-SNR/babble 明确失败；
- MCRA + DD-Wiener 更保守，增强全量上正 SI-SDRi 比例更高；
- 在固定 Whisper 的 100 条 development 子集上，MCRA 与 noisy 持平，RNNoise 明显更差；
- v2 能限制检测到的灾难性重复，却尚未超过 noisy；
- 全量 ASR 方向预计相似，但具体 WER 和显著性仍未知。

不能说：

- “RNNoise 普遍提高 ASR”；
- “100 条已经代表 12,396 条”；
- “oracle 4.57% 可以部署”；
- “official test 是完全盲测”；
- “SI-SDR/STOI 改善证明 WER 必然改善”；
- “一个 fixed Whisper 结果代表所有 ASR 后端”。
