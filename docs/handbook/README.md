# 实时语音增强项目手册

最后更新：2026-08-14

本手册用于项目复现、结果解释和面试准备。内容压缩为五章：

1. [项目、数据与贡献边界](01_项目数据与贡献边界.md)
2. [算法架构与实时工程](02_算法架构与实时工程.md)
3. [代码运行与复现](03_代码运行与复现.md)
4. [实验结果与 ASR 分析](04_实验结果与ASR分析.md)
5. [面试问答与公式速查](05_面试问答与公式速查.md)

阅读时必须区分以下结论层级：

- **增强全量结果**：validation 3,224 对和 official test 824 对已经完成；
- **ASR 诊断结果**：development balanced 100 个 utterance、四路 400 个输入；
- **ASR 主验证结果**：validation 分层 320 个 utterance、四路 1,280 个输入，覆盖全部
  8 speaker × 10 noise × 4 SNR；
- **ASR 全量结果**：development 8,348 和 validation 3,224 未运行；当前证据满足 RNNoise
  方向性停止条件，暂不扩样；
- **R3 的定位**：默认听感 Demo，不是 ASR 默认前端，也不是全场景最优算法；
- **v1 的定位**：固定后端的科研评测，真实失败不回退；
- **v2 的定位**：reference-free 灾难保护，不覆盖 v1；validation selective WER 仍未超过
  paired noisy，且存在 `0.31%` abstention；
- **router 的优先级**：RNNoise/noisy oracle 只比 noisy 低 `0.86 pp`，理论收益上限偏小，
  当前不投入复杂 router；
- **DNSMOS 的定位**：validation 320 四路 1,280 条已完成；它确认 RNNoise 显著改善背景和
  整体感知分，但逐条 delta 几乎不能预测 ASR 词错误变化，不替代 WER 或人工盲听；
- **official test 边界**：项目早期使用过其中的样本，不能称完全盲测。

当前最重要的结论是：

> RNNoise 在低 SNR 非语音型噪声上具有较强的感知降噪能力，但 high-SNR、babble 和
> 固定 Whisper ASR 暴露了明显的过处理风险。是否加入增强前端必须由同一 ASR 后端上的
> paired WER 决定，不能由 SI-SDR、STOI 或听感单独决定；当前 ASR 默认保留 raw/noisy，
> router 只作为未来备选，不是近期主线。

证据位置：

```text
冻结配置：configs/
小型结果和审计：outputs/、data/manifests/
完整数据与大体积产物：/Volumes/T7/ProjectData/realtime_speech_enhancement/
文档图：docs/figures/*.png
```
