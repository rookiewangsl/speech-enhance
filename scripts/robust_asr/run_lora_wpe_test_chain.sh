#!/usr/bin/env bash
set -euo pipefail

data_root="${ROBUST_ASR_DATA_ROOT:?ROBUST_ASR_DATA_ROOT is required}"
python_bin="${ROBUST_ASR_PYTHON:-python3}"

"${python_bin}" scripts/robust_asr/validate_final_test_lock.py \
  --data-root "${data_root}"

"${python_bin}" scripts/robust_asr/run_frozen_whisper_baseline.py \
  --data-root "${data_root}" \
  --limit 1000 \
  --device cuda \
  --local-files-only \
  --frontends raw m_wpe_10 \
  --rt60 0.2 0.4 0.6 0.8 1.0 \
  --rir-split test \
  --manifest-name aishell1_test_reverb.jsonl \
  --output-name w0_whisper_test_reverb_test_1000utt_raw_mwpe_v1.jsonl \
  --checkpoint-every 20 \
  --progress-every 20

for mode in clean mct; do
  run="formal_${mode}_lora_20h_encoder_decoder_qv_v1"
  adapter="${data_root}/runs/${run}/checkpoints/epoch_003"
  output_name="${mode}_lora_whisper_test_reverb_test_1000utt_raw_mwpe_v1.jsonl"
  "${python_bin}" scripts/robust_asr/run_lora_whisper_baseline.py \
    --data-root "${data_root}" \
    --adapter-path "${adapter}" \
    --output-name "${output_name}" \
    --manifest-name aishell1_test_reverb.jsonl \
    --rir-split test \
    --limit 1000 \
    --device cuda \
    --local-files-only \
    --frontends raw m_wpe_10 \
    --rt60 0.2 0.4 0.6 0.8 1.0 \
    --checkpoint-every 20 \
    --progress-every 20
done

"${python_bin}" scripts/robust_asr/summarize_lora_wpe_interaction.py \
  --data-root "${data_root}" \
  --split test \
  --output-name lora_wpe_interaction_test_1000utt_v1.json \
  --bootstrap-draws 10000 \
  --seed 2026

echo "All frozen W0/Clean-LoRA/MCT-LoRA Raw×M-WPE test evaluations completed."
