#!/usr/bin/env bash
set -euo pipefail

data_root="${ROBUST_ASR_DATA_ROOT:?ROBUST_ASR_DATA_ROOT is required}"
python_bin="${ROBUST_ASR_PYTHON:-python3}"

"${python_bin}" scripts/robust_asr/validate_paraformer_test_lock.py \
  --data-root "${data_root}"

"${python_bin}" scripts/robust_asr/run_frozen_paraformer_baseline.py \
  --data-root "${data_root}" \
  --limit 500 \
  --device cuda \
  --cpu-threads 32 \
  --frontends raw m_wpe_10 \
  --rt60 0.2 0.4 0.6 0.8 1.0 \
  --rir-split test \
  --manifest-name aishell1_test_reverb.jsonl \
  --output-name frozen_paraformer_test_reverb_test_500utt_raw_mwpe_v1.jsonl \
  --checkpoint-every 20 \
  --progress-every 20

echo "Frozen Paraformer Raw×M-WPE held-out test completed."
