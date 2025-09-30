#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=agentS-long
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=1-12:00:00

export VLLM_WORKER_MULTIPROC_METHOD=spawn

set -x -e

source ~/.bashrc
cd ~/Projects/open-r1
source .venv/bin/activate

MODEL_ID=$1
TASK_NAME=$2
TASK=$3
CONFIG_FILE=$4
NUM_GPUS=$(nvidia-smi -L | wc -l)

# ---- Parse config file using Python ----
read_yaml () {
python << EOF
import yaml, sys, json
with open("$CONFIG_FILE") as f:
    cfg = yaml.safe_load(f)
val = cfg
# Use the arguments passed to the bash function
keys = "$*".split()
for k in keys:
    if val is None:
        break
    val = val.get(k, None)
# Print scalars directly, JSON for dicts/lists
if isinstance(val, (dict, list)):
    print(json.dumps(val))
else:
    print(val)
EOF
}

MODEL_REVISION=$(read_yaml model_parameters revision)

LM_EVAL_REPO_ID="alexmaraval/open-r1-eval-leaderboard"
DETAILS_REPO_ID="alexmaraval/details-$MODEL_ID"
OUTPUT_DIR="eval_results/$MODEL_ID/$MODEL_REVISION/$TASK_NAME"
ACCELERATE_USE_DEEPSPEED=false

echo "Running lighteval script ..."
echo "Eval results will be saved to $OUTPUT_DIR"
lighteval vllm "$CONFIG_FILE" $TASK \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

OUTPUT_FILEPATHS=$(find $OUTPUT_DIR/results/ -type f \( -name "*.json" \))
for filepath in $OUTPUT_FILEPATHS; do
    echo "Uploading $filepath to Hugging Face Hub..."
    filename=$(basename -- "$filepath")
    for attempt in {1..20}; do
        if hf upload --repo-type space --private $LM_EVAL_REPO_ID $filepath $OUTPUT_DIR/$filename; then
            echo "Upload succeeded for $filepath"
            break
        else
            echo "Upload failed for $filepath. Attempt $attempt of 20. Retrying in 5 seconds..."
            sleep 5
        fi
    done
done

echo "Uploading details to Hugging Face Hub..."
DETAILS_FILEPATHS=$(find $OUTPUT_DIR/details/ -type f \( -name "*.parquet" \))
echo "DETAILS_FILEPATHS: $DETAILS_FILEPATHS"
TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
python scripts/upload_details.py --data_files "$DETAILS_FILEPATHS" --hub_repo_id "$DETAILS_REPO_ID" \
  --config_name "${MODEL_REVISION}.${TASK_NAME}.${TIMESTAMP}"

echo "Done!"