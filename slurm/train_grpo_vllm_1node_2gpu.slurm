#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:2
#SBATCH --job-name=open_r1
#SBATCH --partition=agentS-xlong
#SBATCH --time=5-00:00:00
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --requeue

if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch slurm/train.slurm [options]"
  echo "Options:"
  echo "  --model MODEL            Model name"
  echo "  --task TASK              Task name (e.g. sft, grpo)"
  echo "  --config SUFFIX          Configuration suffix (e.g. demo, v00.00)"
  echo "  --args \"ARGS\"          Optional arguments to pass to the training script"
  exit 0
fi

set -x -e
eval "$(conda shell.bash hook)"
cd ~/Projects/open-r1/
source .venv/bin/activate
START_TIME=$(date +%s)
echo "START TIME: $(date)"

# Default values
MODEL=""
TASK=""
CONFIG_SUFFIX=""
OPTIONAL_ARGS=""
VLLM_HOST="localhost"
VLLM_PORT=8010

export WANDB_PROJECT="open-r1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"; shift 2 ;;
    --task)
      TASK="$2"; shift 2 ;;
    --config)
      CONFIG_SUFFIX="$2"; shift 2 ;;
    --args)
      OPTIONAL_ARGS="$2"; shift 2 ;;
    --vllm_host)
      VLLM_HOST="$2"; shift 2 ;;
    --vllm_port)
      VLLM_PORT="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$TASK" || -z "$CONFIG_SUFFIX" ]]; then
  echo "Error: Missing required arguments"
  echo "Run with --help for usage information"
  exit 1
fi

CONFIG_FILE=recipes/$MODEL/$TASK/config_$CONFIG_SUFFIX.yaml
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')

# Allow override via --args
IFS=' ' read -ra ARGS <<< "$OPTIONAL_ARGS"
for arg in "${ARGS[@]}"; do
    if [[ "$arg" == --gradient_accumulation_steps=* ]]; then
        GRAD_ACC_STEPS="${arg#*=}"
        break
    fi
done
echo "Gradient accumulation steps: $GRAD_ACC_STEPS"

MODEL=$(grep 'model_name_or_path:' $CONFIG_FILE | awk '{print $2}')
REVISION=$(grep 'model_revision:' $CONFIG_FILE | head -n 1 | awk '{print $2}')

# ---------- vLLM ----------
echo "Starting vLLM server on GPU:1..."
CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
  --model $MODEL --revision $REVISION \
  --enforce-eager False --trust_remote_code True \
  --host $VLLM_HOST --port $VLLM_PORT \
  --log-level warning &

OPTIONAL_ARGS="$OPTIONAL_ARGS --vllm_server_host localhost --vllm_server_port 8010"

echo "Waiting for vLLM server to start..."
until nc -z $VLLM_HOST $VLLM_PORT; do sleep 5; done
echo "vLLM is up!"

# Training on GPU:0
CUDA_VISIBLE_DEVICES=0 python src/open_r1/$TASK.py \
    --config $CONFIG_FILE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    $OPTIONAL_ARGS

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"