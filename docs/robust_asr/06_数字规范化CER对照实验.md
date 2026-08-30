# 数字规范化 CER 对照实验

## 1. 实验定位

状态：**已完成**。本实验只复用封存 test 的现有转写进行只读重评分，不重新训练、不重新推理，也不改写
原始 JSONL。预注册中文 CER 继续作为项目主指标；数字规范化 CER 是用于区分声学错误与输出书写形式的
辅助指标。

要回答四个问题：

1. 原始 Whisper W0 的绝对 CER 有多少来自阿拉伯数字与中文读法不一致？
2. W0→Clean-LoRA 的收益在消除数字书写形式差异后是否仍存在？
3. WPE 和 MCT 的混响结论是否依赖原始数字评分方式？
4. Whisper 与 Paraformer 的中文差距能否由数字格式解释？

## 2. 冻结的评分对照

所有规则都对 reference 和 hypothesis 对称应用。只有最后一列允许逐句搜索，不能替代正式辅助指标。

| 评分 | 规则 | 角色 |
|---|---|---|
| Formal | 原预注册 NFKC/繁简/字符过滤；保留阿拉伯数字 | 主指标，完全不变 |
| Contextual | `%→百分之`；四位年份逐位；其他整数按基数；小数为“整数+点+逐位小数” | 固定的主要辅助指标 |
| Digit-by-digit | 所有阿拉伯数字逐位读取，百分数仍补“百分之” | 规则敏感性对照 |
| Diagnostic bound | 每句在逐位、基数和上述固定规则候选中取最小编辑距离 | 诊断下界，不用于模型排名 |

固定规则示例：

| 原文本 | Contextual 输出 |
|---|---|
| `46%` | `百分之四十六` |
| `2014年` | `二零一四年` |
| `15.107元` | `十五点一零七元` |
| `30-40万元` | `三十至四十万元` |
| `1,500元` | `一千五百元` |

“10”可能读作“十”或“一零”，固定规则不查看参考答案决定读法；这正是 Digit-by-digit 和诊断下界存在的
原因。

## 3. 数据与对照矩阵

| 标签 | 数据 | 条件 | 规模 |
|---|---|---|---:|
| W0 | Whisper-small 冻结模型 | Clean、Raw、M-WPE；五档 RT60 | 11,000 行 |
| Clean-LoRA | 20 h clean LoRA | 同上 | 11,000 行 |
| MCT-LoRA | 20 h 50/50 clean/reverb LoRA | 同上 | 11,000 行 |
| Paraformer | 冻结中文交叉模型 | 同上 | 5,500 行 |

Robust 汇总 RT60 `0.4/0.6/0.8/1.0 s`，Heavy 汇总 `0.8/1.0 s`。成对置信区间使用
10,000 次 bootstrap、seed `2026`；同一语句的多个 RT60 条件先聚合为一个 cluster，避免把四档条件误当作
相互独立语句。

## 4. 绝对 CER 结果

### 4.1 固定 Contextual 规则

| 模型 | Clean Formal | Clean Num | Robust Raw Formal | Robust Raw Num | Robust M-WPE Formal | Robust M-WPE Num |
|---|---:|---:|---:|---:|---:|---:|
| W0 | 14.08% | 10.95% | 23.61% | 20.82% | 15.39% | 12.47% |
| Clean-LoRA | 7.76% | 7.76% | 15.33% | 15.32% | 9.18% | 9.15% |
| MCT-LoRA | 7.83% | 7.83% | 14.17% | 14.16% | 9.08% | 9.08% |
| Paraformer | 2.33% | 2.33% | 3.32% | 3.32% | 2.28% | 2.28% |

数字格式主要影响 W0。Clean-LoRA、MCT-LoRA 和 Paraformer 的 clean hypothesis 都没有阿拉伯数字；
Robust 汇总中也只有 0–4 行，因此其 CER 几乎不变。

### 4.2 W0 规则敏感性

| 评分 | Clean | Robust Raw | Robust M-WPE |
|---|---:|---:|---:|
| Formal | 14.08% | 23.61% | 15.39% |
| Contextual | 10.95% | 20.82% | 12.47% |
| Digit-by-digit | 11.39% | 21.21% | 12.86% |
| Diagnostic bound | 10.60% | 20.48% | 12.12% |

两种不择优的固定规则相差 `0.38–0.45 pp`；逐句最优下界再低约 `0.34–0.35 pp`。因此绝对
number-aware CER 对读法策略有小幅敏感性，但远小于 W0 与 LoRA/Paraformer 的剩余差距。

### 4.3 W0 Clean 的受影响子集

| 子集 | 语句数 | Formal CER | Contextual CER |
|---|---:|---:|---:|
| hypothesis 含阿拉伯数字 | 141 | 31.82% | 11.59% |
| hypothesis 不含阿拉伯数字 | 859 | 10.83% | 10.83% |

在 141 条数字输出中，Contextual 规则使 136 条改善、2 条恶化、3 条不变。少量恶化说明固定读法仍有
语境歧义或规则未覆盖，不能把它包装成无误差的完整中文 ITN。

## 5. 成对效应是否改变

所有数值均为 candidate−baseline；负值表示 CER 改善。

| 对照 | Formal ΔCER（95% CI） | Contextual ΔCER（95% CI） | 判断 |
|---|---:|---:|---|
| Clean-LoRA−W0，Clean | −6.32 `[−7.16,−5.50]` | −3.18 `[−3.75,−2.62]` | 收益缩小但仍显著 |
| Clean-LoRA−W0，Robust Raw | −8.28 `[−9.23,−7.41]` | −5.50 `[−6.26,−4.83]` | 收益缩小但仍显著 |
| W0 M-WPE−Raw，Robust | −8.22 `[−9.02,−7.51]` | −8.35 `[−9.16,−7.65]` | WPE 结论不变 |
| Clean-LoRA M-WPE−Raw，Robust | −6.15 `[−6.63,−5.69]` | −6.16 `[−6.64,−5.70]` | WPE 结论不变 |
| MCT-LoRA M-WPE−Raw，Robust | −5.09 `[−5.54,−4.66]` | −5.08 `[−5.53,−4.65]` | WPE 结论不变 |
| MCT-LoRA−Clean-LoRA，Robust Raw | −1.16 `[−1.41,−0.90]` | −1.16 `[−1.42,−0.91]` | MCT 结论不变 |
| Paraformer M-WPE−Raw，Robust | −1.03 `[−1.30,−0.78]` | −1.03 `[−1.30,−0.78]` | 跨模型结论不变 |

## 6. 结论与项目叙事修正

1. **W0 的绝对中文 CER 被数字格式明显抬高。** Clean 从 `14.08%` 修正为固定规则下的
   `10.95%`，不能把 `14.08%` 全部称为声学识别失败。
2. **Clean-LoRA 不只是声学/语言适配，也学会了标注输出风格。** W0→Clean-LoRA 的 clean
   主 CER 收益为 `6.32 pp`；数字规范化后剩余 `3.18 pp`。可把前后差额 `3.14 pp` 作为评分层面的
   粗略分解，但不能宣称为严格因果分解。
3. **LoRA 仍有独立价值。** 数字规范化后 clean 与 Robust Raw 的改善区间仍完全低于零，所以收益并非
   全来自把阿拉伯数字改成中文数字。
4. **WPE 是最稳定的混响主结论。** 数字规范化后 W0 的 Robust 收益反而从 `8.22 pp` 变为
   `8.35 pp`，Clean-LoRA、MCT-LoRA 和 Paraformer 上也不变。
5. **MCT 的小幅独立收益不受数字格式影响。** Raw 上仍为约 `1.16 pp`，其定位继续是没有多通道
   WPE 输入时的单通道鲁棒化方案。
6. **Paraformer 的中文优势不是评分假象。** 即使把 W0 clean 修正为 `10.95%`，仍显著高于
   Paraformer 的 `2.33%`。

简历仍使用预注册主 CER；面试中若被问到 Whisper 中文基线偏高，应主动补充 number-aware CER，并说明
自己通过固定规则、敏感性策略、诊断下界和成对 bootstrap 区分了输出格式与真实鲁棒性收益。

## 7. 复现

单文件重评分：

```bash
python scripts/robust_asr/rescore_whisper_outputs.py \
  --data-root /home/slwang/data/realtime_speech_enhancement/robust_asr \
  --input-name w0_whisper_test_reverb_test_1000utt_raw_mwpe_v1.jsonl \
  --output-name w0_whisper_test_reverb_test_1000utt_raw_mwpe_v1.number_normalization_v2.json \
  --quiet
```

四模型绝对结果由 `scripts/robust_asr/summarize_number_normalization.py` 聚合；成对区间由
`scripts/robust_asr/compare_number_normalization.py` 计算。正式证据位于服务器数据根目录：

- `outputs/number_normalization_formal_test_v1.json`；
- `outputs/number_normalization_paired_contrasts_v1.json`；
- 四个 `outputs/*.number_normalization_v2.json`。

代码单元测试覆盖百分比、年份、小数、范围、千分位、固定读法歧义、RT60 cluster 完整性和汇总逻辑。
