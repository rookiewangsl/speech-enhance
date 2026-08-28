#!/usr/bin/env bash
set -euo pipefail

data_root="${ROBUST_ASR_DATA_ROOT:?ROBUST_ASR_DATA_ROOT is required}"
python_bin="${ROBUST_ASR_PYTHON:-python3}"
workers="${ROBUST_ASR_WORKERS:-32}"
prefetch="${ROBUST_ASR_PREFETCH:-4}"

target="encoder_decoder_qv"
train_hours=20
epochs=3

runs=(
  "formal_clean_lora_20h_encoder_decoder_qv_v1:clean"
  "formal_mct_lora_20h_encoder_decoder_qv_v1:mct"
)

for run in "${runs[@]}"; do
  experiment="${run%%:*}"
  mode="${run#*:}"
  run_dir="${data_root}/runs/${experiment}"
  summary="${run_dir}/training_summary.json"

  if [[ -f "${summary}" ]]; then
    if grep -Eq '"status"[[:space:]]*:[[:space:]]*"SUCCESS"' "${summary}"; then
      echo "Formal run already complete: ${experiment}"
      continue
    fi
    echo "Refusing to skip non-successful completed run: ${summary}" >&2
    exit 1
  fi

  command=(
    "${python_bin}" scripts/robust_asr/train_whisper_lora.py
    --data-root "${data_root}"
    --experiment-id "${experiment}"
    --mode "${mode}"
    --lora-target "${target}"
    --train-hours "${train_hours}"
    --epochs "${epochs}"
    --num-workers "${workers}"
    --prefetch-factor "${prefetch}"
    --console-every 20
    --local-files-only
  )
  if [[ -f "${run_dir}/latest_state.pt" ]]; then
    command+=(--resume-from "${run_dir}/latest_state.pt")
    echo "Resuming formal run from the last completed epoch: ${experiment}"
  else
    echo "Starting formal run: ${experiment}"
  fi

  "${command[@]}"
done

echo "All formal LoRA runs completed."
