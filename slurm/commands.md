# SFT

```shell
sbatch \
  --job-name="openPangu-distill" \
  --nodes=1 \
  --partition="agentS-xlong" \
  --time="5-00:00:00" \
  --gres="gpu:h200:2" \
  --ntasks-per-node=1 \
  slurm/train.slurm \
      --model "openPangu-Embedded-1B" \
      --task "sft" \
      --config "distill" \
      --accelerator "zero2" \
      --dp 2 \
      --args "--run_name=openPangu-distill"
```

Or from an interactive session

```bash
export WANDB_PROJECT="open-r1"
export NCCL_ASYNC_ERROR_HANDLING=1
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2_debug.yaml \
    --gradient_accumulation_steps 32 \
    src/open_r1/sft.py \
        --config recipes/openPangu-Embedded-1B/sft/config_distill_debug.yaml \
        --run_name=debug-openPangu-distill
```

# GRPO

**Note** that we need 2 nodes for GRPO training, one node for the trainer/policy and one node for the vLLM server. In
this particular example, node will have 2GPUs and DP=2.

```shell
sbatch \
  --job-name="openPangu-grpo-math" \
  --nodes=2 \
  --partition="agentS-xlong" \
  --time="5-00:00:00" \
  --gres="gpu:h200:2" \
  --ntasks-per-node=1 \
  slurm/train.slurm \
      --model "openPangu-Embedded-1B" \
      --task "grpo" \
      --config "math" \
      --accelerator "zero2" \
      --dp 2 \
      --args "--run_name=openPangu-grpo-math"
```

# Evaluate

### Launch jobs with `sbatch`

```shell
sbatch --job-name eval-math500 slurm/evaluate.slurm \
  "math_500" "lighteval|math_500|0|0" \
  "FreedomIntelligence/openPangu-Embedded-1B" \
  "main" False True
```

```shell
sbatch --job-name eval-aime24 slurm/evaluate.slurm \
  "aime24" "lighteval|aime24|0|0" \
  "FreedomIntelligence/openPangu-Embedded-1B" \
  "main" False True
```

```shell
sbatch --job-name eval-aime25 slurm/evaluate.slurm \
  "aime25" "lighteval|aime25|0|0" \
  "FreedomIntelligence/openPangu-Embedded-1B" \
  "main" False True
```

```shell
sbatch --job-name eval-gsm8k slurm/evaluate.slurm \
  "lcb:codegeneration" "lighteval|gsm8k|0|0" \
  "FreedomIntelligence/openPangu-Embedded-1B" \
  "main" False True
```

```shell
sbatch --job-name eval-gpqa-diamond slurm/evaluate.slurm \
  "gpqa:diamond" "lighteval|gpqa:diamond|0|0" \
  "FreedomIntelligence/openPangu-Embedded-1B" \
  "main" False True
```

```shell
sbatch --job-name eval-mmlu slurm/evaluate.slurm \
  "mmlu" "original|mmlu|0|0" \
  "FreedomIntelligence/openPangu-Embedded-1B" \
  "main" False True
```

```shell
sbatch --job-name eval-lcb slurm/evaluate.slurm \
  "lcb:codegeneration" "lcb:codegeneration|aime24|0|0" \
  "FreedomIntelligence/openPangu-Embedded-1B" \
  "main" False True
```

### Run directly in a dev session

If using vLLM

```bash
MODEL_ID="FreedomIntelligence/openPangu-Embedded-1B"
MODEL_REVISION="main"

declare -A TASKS_MAP=(
  ["gsm8k"]="lighteval|gsm8k|0|0"
  ["math_500"]="lighteval|math_500|0|0"
  ["gpqa_diamond"]="lighteval|gpqa:diamond|0|0"
  ["aime24"]="lighteval|aime24|0|0"
  ["aime25"]="lighteval|aime25|0|0"
  ["mmlu"]="original|mmlu|0|0"
#  ["lcb_codegen"]="extended|lcb:codegeneration|0|0"
)

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="lighteval"
ACCELERATE_USE_DEEPSPEED=false
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
LM_EVAL_REPO_ID="alexmaraval/open-r1-eval-leaderboard"
MODEL_NAME=$(echo $MODEL_ID | sed 's/\//_/g') # replaces / with _
DETAILS_REPO_ID="alexmaraval/details-$MODEL_NAME"

for TASK_NAME in "${!TASKS_MAP[@]}"; do
    TASKS="${TASKS_MAP[$TASK_NAME]}"
    OUTPUT_DIR="eval_results/$MODEL_NAME/$MODEL_REVISION/$TASK_NAME"

    echo "Running lighteval script for $TASK_NAME ..."
    echo "Eval results will be saved to $OUTPUT_DIR"

    lighteval vllm recipes/openPangu-Embedded-1B/evaluate/config_eval_vllm.yaml $TASKS \
        --use-chat-template \
        --output-dir $OUTPUT_DIR \
        --save-details
done
```

If using Accelerate, replace the last part with

```bash
lighteval accelerate recipes/openPangu-Embedded-1B/evaluate/config_eval_accelerate.yaml $TASKS \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details
```