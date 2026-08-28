#!/usr/bin/env bash
set -uo pipefail

data_root="${ROBUST_ASR_DATA_ROOT:?ROBUST_ASR_DATA_ROOT is required}"
python_bin="${ROBUST_ASR_PYTHON:-python3}"
workers="${ROBUST_ASR_WORKERS:-32}"
prefetch="${ROBUST_ASR_PREFETCH:-4}"

benchmark_name="benchmark_mct_encoder_qv_b2a8_100step_v1"
benchmark_summary="${data_root}/runs/${benchmark_name}/benchmark_summary.json"
if [[ -f "${benchmark_summary}" ]]; then
  echo "Hardware benchmark already complete: ${benchmark_summary}"
else
  "${python_bin}" scripts/robust_asr/benchmark_whisper_lora.py \
    --data-root "${data_root}" \
    --mode mct \
    --lora-target encoder_qv \
    --optimizer-steps 100 \
    --num-workers "${workers}" \
    --prefetch-factor "${prefetch}" \
    --console-every 20 \
    --output-name "${benchmark_name}" \
    --local-files-only
fi

pilot_failed=0
for target in encoder_qv encoder_decoder_qv; do
  experiment="pilot_mct_5h_${target}_500step_v1"
  run_dir="${data_root}/runs/${experiment}"
  if [[ -f "${run_dir}/training_summary.json" ]]; then
    echo "Pilot already complete: ${experiment}"
    continue
  fi
  resume_args=()
  if [[ -f "${run_dir}/latest_state.pt" ]]; then
    resume_args=(--resume-from "${run_dir}/latest_state.pt")
    echo "Resuming pilot from completed epoch: ${experiment}"
  fi
  if ! "${python_bin}" scripts/robust_asr/train_whisper_lora.py \
    --data-root "${data_root}" \
    --experiment-id "${experiment}" \
    --mode mct \
    --lora-target "${target}" \
    --train-hours 5 \
    --epochs 2 \
    --maximum-optimizer-steps 500 \
    --num-workers "${workers}" \
    --prefetch-factor "${prefetch}" \
    --console-every 20 \
    --local-files-only \
    "${resume_args[@]}"; then
    echo "Pilot failed: ${experiment}" >&2
    pilot_failed=1
  fi
done

if [[ "${pilot_failed}" -ne 0 ]]; then
  exit 1
fi
echo "All LoRA pilot runs completed."
