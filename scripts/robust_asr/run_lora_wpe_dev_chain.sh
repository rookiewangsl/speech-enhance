#!/usr/bin/env bash
set -euo pipefail

data_root="${ROBUST_ASR_DATA_ROOT:?ROBUST_ASR_DATA_ROOT is required}"
python_bin="${ROBUST_ASR_PYTHON:-python3}"

w0_source="${data_root}/outputs/frozen_whisper_dev_model_dev_1000utt_v1.jsonl"
w0_output_name="w0_whisper_dev_model_dev_1000utt_raw_mwpe_v1.jsonl"
w0_output="${data_root}/outputs/${w0_output_name}"
if [[ ! -f "${w0_output}" ]]; then
  cp "${w0_source}" "${w0_output}"
fi
"${python_bin}" scripts/robust_asr/run_frozen_whisper_baseline.py \
  --data-root "${data_root}" \
  --limit 1000 \
  --device cuda \
  --local-files-only \
  --frontends raw m_wpe_10 \
  --rt60 0.2 0.4 0.6 0.8 1.0 \
  --rir-split dev \
  --manifest-name aishell1_dev_model.jsonl \
  --output-name "${w0_output_name}" \
  --checkpoint-every 20 \
  --progress-every 20

for mode in clean mct; do
  run="formal_${mode}_lora_20h_encoder_decoder_qv_v1"
  run_dir="${data_root}/runs/${run}"
  summary="${run_dir}/training_summary.json"
  if [[ ! -f "${summary}" ]] || ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"SUCCESS"' "${summary}"; then
    echo "Formal LoRA run is not complete: ${run}" >&2
    exit 1
  fi
  adapter="${run_dir}/checkpoints/epoch_003"
  seed_predictions="${run_dir}/dev/epoch_003/predictions.jsonl"
  output_name="${mode}_lora_whisper_dev_model_dev_1000utt_raw_mwpe_v1.jsonl"
  "${python_bin}" scripts/robust_asr/run_lora_whisper_baseline.py \
    --data-root "${data_root}" \
    --adapter-path "${adapter}" \
    --seed-predictions "${seed_predictions}" \
    --output-name "${output_name}" \
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
  --output-name lora_wpe_interaction_dev_1000utt_v1.json \
  --bootstrap-draws 10000 \
  --seed 2026

echo "All W0/Clean-LoRA/MCT-LoRA Raw×M-WPE dev evaluations completed."
