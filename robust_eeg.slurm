#!/bin/bash
#SBATCH --account=rrg-cevacu
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --array=0-35
#SBATCH --job-name=robust-eeg
#SBATCH --output=/scratch/shyamal/physionet-2026/logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MNE_DONTWRITE_HOME=true
mkdir -p /scratch/shyamal/physionet-2026/results/robust

# Grid: model {cnn,rescnn} x channel_dropout {0.0,0.15} x seed {0,1,2} x fold {1,2,3}
i="$SLURM_ARRAY_TASK_ID"
models=(cnn rescnn)
model="${models[$((i / 18))]}"
rest=$((i % 18))
cds=(0.0 0.15)
cd_rate="${cds[$((rest / 9))]}"
rest2=$((rest % 9))
seed=$((rest2 / 3))
fold=$((rest2 % 3 + 1))

out="/scratch/shyamal/physionet-2026/results/robust/${model}_cd${cd_rate}_s${seed}_f${fold}"
if [[ -f "$out.json" ]]; then
  echo "already done: $out"
  exit 0
fi

env -u PYTHONPATH .venv/bin/python benchmark_robust_eeg.py \
  --model-name "$model" \
  --channel-dropout "$cd_rate" \
  --train-seed "$seed" \
  --fold "$fold" \
  --epochs 50 \
  --batch-size 512 \
  --num-workers 4 \
  --output-base "$out"
