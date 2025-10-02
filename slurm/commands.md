# SFT

```shell
#model="openPangu-Embedded-1B"
model="openPangu-Embedded-7B"
#model="Qwen2.5-1.5B-Instruct"
task="sft"
config="distill"
accelerator="zero2-2gpu"

job_name="${model}-${config}"

sbatch --job-name="$job_name" \
  slurm/train.slurm \
      --model "$model" \
      --task "$task" \
      --config "$config" \
      --accelerator "$accelerator" \
      --args "--run_name=${job_name}"
```

Or from an interactive session

```bash
export WANDB_PROJECT="open-r1"
export NCCL_ASYNC_ERROR_HANDLING=1
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2-2gpu.yaml \
    --gradient_accumulation_steps 32 \
    src/open_r1/sft.py \
        --config recipes/openPangu-Embedded-1B/sft/config_distill_debug.yaml \
        --run_name=debug-openPangu-distill
        
python src/open_r1/sft.py \
  --config recipes/Qwen2.5-1.5B-Instruct/sft/config_distill_debug.yaml \
  --run_name=debug-Qwen2.5-1.5B-Instruct-distill
```

# GRPO

**Note** that we need 2 nodes for GRPO training, one node for the trainer/policy and one node for the vLLM server. In
this particular example, node will have 2GPUs and DP=2.

```shell
#model="openPangu-Embedded-1B"
model="openPangu-Embedded-7B"
#model="Qwen2.5-1.5B-Instruct"
task="grpo"
config="math"
#config="gsm8k"
accelerator="zero2-2gpu"
dp=2

job_name="${model}-${task}-${config}"
nodes=2
partition="agentS-long"
time="1-12:00:00"
#partition="agentS-xlong"
#time="5-00:00:00"
gres="gpu:h200:2"

sbatch \
  --job-name="$job_name" \
  --nodes="$nodes" \
  --partition="$partition" \
  --time="$time" \
  --gres="$gres" \
  slurm/train.slurm \
      --model "$model" \
      --task "$task" \
      --config "$config" \
      --accelerator "$accelerator" \
      --dp "$dp" \
      --args "--run_name=${job_name}"
```

Or to launch on 3 GPUs on one single node with vLLM server, use this command

```shell
task="grpo"
#config="math1k"
#config="gsm8k"
config="math220k"
accelerator="zero2-2gpu"

model="openPangu-Embedded-1B"
#model="openPangu-Embedded-7B"
#model="Qwen2.5-1.5B-Instruct"

job_name="${model}-${task}-${config}"
```

```shell
partition="agentS-xlong"
time="5-00:00:00"
sbatch --job-name="$job_name" \
       --partition="$partition" \
       --time="$time" \
  slurm/train_grpo_vllm_1node.slurm \
    --model "$model" \
    --task "$task" \
    --config "$config" \
    --accelerator "$accelerator" \
    --args "--run_name=${job_name}"
```

```shell
partition="agentS-long"
time="1-12:00:00"
job_name="${model}-${task}-${config}"
sbatch --job-name="$job_name" \
       --partition="$partition" \
       --time="$time" \
  slurm/train_grpo_vllm_1node.slurm \
    --model "$model" \
    --task "$task" \
    --config "$config" \
    --accelerator "$accelerator" \
    --args "--run_name=${job_name}"
```

# Evaluate

### Launch jobs with `sbatch`

```shell
# ------------------------------ FIRST! ------------------------------
# Edit directly `model_name_or_path` and `revision` in the config.yaml
# --------------------------------------------------------------------
#MODEL_ID="openPangu-Embedded-1B"
MODEL_ID="openPangu-Embedded-7B"
#MODEL_ID="Qwen2.5-1.5B-Instruct"
#MODEL_ID="Qwen3-0.6B"
#MODEL_ID="Qwen3-1.7B"
EVAL_CONFIG="recipes/${MODEL_ID}/evaluate/config_eval_vllm.yaml"

sbatch --job-name eval-math500 slurm/evaluate2.slurm \
  "$MODEL_ID" "math_500" "lighteval|math_500|0|0" "$EVAL_CONFIG"

sbatch --job-name eval-aime24 slurm/evaluate2.slurm \
  "$MODEL_ID" "aime24" "lighteval|aime24|0|0" "$EVAL_CONFIG"

sbatch --job-name eval-aime25 slurm/evaluate2.slurm \
  "$MODEL_ID" "aime25" "lighteval|aime25|0|0" "$EVAL_CONFIG"

sbatch --job-name eval-gsm8k slurm/evaluate2.slurm \
  "$MODEL_ID" "gsm8k" "lighteval|gsm8k|0|0" "$EVAL_CONFIG"

sbatch --job-name eval-gpqa-diamond slurm/evaluate2.slurm \
  "$MODEL_ID" "gpqa_diamond" "lighteval|gpqa:diamond|0|0" "$EVAL_CONFIG"

sbatch --job-name eval-mmlu slurm/evaluate2.slurm \
  "$MODEL_ID" "mmlu" "original|mmlu|0|0" "$EVAL_CONFIG"

#sbatch --job-name eval-lcb slurm/evaluate2.slurm \
#  "$MODEL_ID" "lcb_codegen" "lighteval|lcb:codegeneration|0|0" "$EVAL_CONFIG"
```

### Run directly in a dev session

If using vLLM

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="lighteval"
ACCELERATE_USE_DEEPSPEED=false
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
LM_EVAL_REPO_ID="alexmaraval/open-r1-eval-leaderboard"
DETAILS_REPO_ID="alexmaraval/details-$MODEL_ID"

# Edit directly `model_name_or_path` and `revision` in the config.yaml
#EVAL_CONFIG=recipes/openPangu-Embedded-1B/evaluate/config_eval_vllm.yaml
EVAL_CONFIG=recipes/openPangu-Embedded-7B/evaluate/config_eval_vllm.yaml
#MODEL_ID="openPangu-Embedded-1B"
MODEL_ID="openPangu-Embedded-7B"
MODEL_REVISION="main"
#MODEL_REVISION="grpo_gsm8k"
#TASK="lighteval|gsm8k|0|0"
#TASK_NAME="gsm8k"
#TASK="lighteval|math_500|0|0"
#TASK_NAME="math_500"
TASK="lighteval|aime24|0|0"
TASK_NAME="aime24"
OUTPUT_DIR="eval_results/$MODEL_ID/$MODEL_REVISION/$TASK_NAME"

echo "Running lighteval script for $TASK_NAME ..."
echo "Eval results will be saved to $OUTPUT_DIR"

lighteval vllm $EVAL_CONFIG $TASK \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details
```