# Changelog

**[v0.1]** Setup + openPangu-Embedded-1B support

- added configs for openPangu-Embedded-1B
- fixed `train.slurm` and `evaluate.slurm` as well as configs to adapt to our cluster
- setup evaluation results scripts
- fix SFT checkpoint saving
- **PATCH** `GRPOTrainer` into [PatchedGRPOTrainer](src/open_r1/custom_trainers/patched_grpo_trainer.py) to support
  passing a reference model so that we can use KL divergence regularization with `openPangu-Embedded-1B`.
- added training and eval configs for other small LLMs and simplify `evaluate.slurm` → `evaluate2.slurm`
- add option to train GRPO on 1 node with 2 GPUs for trainer and 1 GPU for vLLM server
- add support for RLOO training

# TODO

- [x] See if I can patch `trl/trainer/grpo_trainer.py` when the reference model is created in a distributed setting, we
  have the cole below where `config = AutoConfig.from_pretrained(model_id)` fails. We would need to e.g. add a
  `ref_model` argument to the constructor.

# Notes

#### Example: aiming for 1k optimization steps in training

- total_samples_per_batch = num_gpus * grad_accumulation_steps * per_device_batch_size = 8 * 32 * 4 = 1024
- unique_prompts_per_batch = total_samples_per_batch / num_generations = 1024 / 16 = 64
- #dataset ~= 16k (8k * 2, for python and cpp)
- global_steps_per_epoch = #dataset / unique_prompts_per_batch = 16k / 64 ~= 250
- epochs_for_1k_steps = 1000/250 = 4 epochs

# Results

### openPangu-Embedded-1B

|              |            Metric | openPangu-Embedded-1B | openPangu-Embedded-1B-Distill | Qwen2.5-1.5B-Instruct |     Qwen3-0.6B |         Qwen3-1.7B |
|--------------|------------------:|----------------------:|------------------------------:|----------------------:|---------------:|-------------------:|
| math500      |  pass@1:1_samples |        58.20 +/- 2.21 |                49.00 +/- 2.24 |        52.80 +/- 2.23 | 71.60 +/- 2.02 | **90.00 +/- 1.34** |
|              |  pass@1:4_samples |        58.40 +/- 1.77 |                48.05 +/- 1.71 |        53.70 +/- 1.84 | 71.30 +/- 1.63 | **90.10 +/- 1.06** |
| gsm8k        |  extractive_match |        64.54 +/- 1.32 |                 62.7 +/- 1.33 |        64.14 +/- 1.32 | 76.88 +/- 1.16 | **87.87 +/- 0.90** |
| aime24       |  pass@1:1_samples |         3.33 +/- 3.33 |                 3.33 +/- 3.33 |         0.00 +/- 0.00 | 13.33 +/- 6.31 |     36.67 +/- 8.95 |
|              |  pass@1:4_samples |         6.67 +/- 2.92 |                 2.50 +/- 1.84 |         3.33 +/- 1.98 | 10.00 +/- 4.42 |     46.67 +/- 7.46 |
|              |  pass@1:8_samples |         7.92 +/- 3.14 |                 3.33 +/- 1.89 |         2.08 +/- 1.05 |  8.33 +/- 3.99 |     43.33 +/- 7.14 |
|              | pass@1:16_samples |         6.46 +/- 2.46 |                 2.92 +/- 1.79 |         2.08 +/- 1.01 |  9.79 +/- 3.59 |     43.75 +/- 6.85 |
|              | pass@1:32_samples |         6.25 +/- 2.45 |                 2.60 +/- 1.73 |         2.50 +/- 1.04 | 10.94 +/- 3.93 |     44.48 +/- 6.49 |
|              | pass@1:64_samples |         6.88 +/- 2.63 |                 2.76 +/- 1.90 |         2.50 +/- 1.08 | 11.51 +/- 3.98 |     44.74 +/- 6.40 |
| aime25       |  pass@1:1_samples |         0.00 +/- 0.00 |                10.00 +/- 5.57 |         0.00 +/- 0.00 | 13.33 +/- 6.31 |     36.67 +/- 8.95 |
|              |  pass@1:4_samples |         5.83 +/- 2.59 |                 6.67 +/- 4.14 |         1.67 +/- 1.16 | 14.17 +/- 5.32 |     35.83 +/- 7.83 |
|              |  pass@1:8_samples |         7.50 +/- 3.32 |                 7.08 +/- 3.58 |         0.83 +/- 0.58 | 14.17 +/- 5.11 |     34.58 +/- 7.45 |
|              | pass@1:16_samples |         8.54 +/- 3.72 |                 5.83 +/- 3.06 |         0.62 +/- 0.35 | 13.75 +/- 4.91 |     35.21 +/- 7.34 |
|              | pass@1:32_samples |         8.33 +/- 3.68 |                 5.94 +/- 3.02 |         1.04 +/- 0.50 | 13.54 +/- 4.74 |     35.00 +/- 7.23 |
|              | pass@1:64_samples |         8.39 +/- 3.68 |                 6.41 +/- 3.09 |         1.15 +/- 0.51 | 13.96 +/- 4.64 |     34.79 +/- 7.10 |
| gpqa-diamond |  pass@1:1_samples |        31.31 +/- 3.30 |                31.82 +/- 3.32 |        25.76 +/- 3.12 | 28.28 +/- 3.21 |     41.41 +/- 3.51 |
|              |  pass@1:4_samples |        35.48 +/- 2.20 |                29.92 +/- 1.82 |        27.90 +/- 1.88 | 27.27 +/- 2.09 |     41.79 +/- 2.82 |
|              |  pass@1:8_samples |        36.36 +/- 1.93 |                30.81 +/- 1.64 |        27.97 +/- 1.66 | 28.47 +/- 2.07 |     40.21 +/- 2.68 |
| mmlu         |               acc |        27.54 +/- 3.32 |                24.81 +/- 3.22 |    **59.70 +/- 3.49** | 23.12 +/- 3.15 |     23.12 +/- 3.15 |

