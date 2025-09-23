# Notes

#### Example: aiming for 1k optimization steps in training

- total_samples_per_batch = num_gpus * grad_accumulation_steps * per_device_batch_size = 8 * 32 * 4 = 1024
- unique_prompts_per_batch = total_samples_per_batch / num_generations = 1024 / 16 = 64
- #dataset ~= 16k (8k * 2, for python and cpp)
- global_steps_per_epoch = #dataset / unique_prompts_per_batch = 16k / 64 ~= 250
- epochs_for_1k_steps = 1000/250 = 4 epochs


# Results

### openPangu-Embedded-1B
| Task                     |Version| Metric                 |Value |  |Stderr|
|--------------------------|------:|------------------------|-----:|--|-----:|
| lighteval:math_500:0     |      2| math_pass@1:1_samples  |0.5920|± |0.0220|
|                          |       | math_pass@1:4_samples  |0.5855|± |0.0175|
| lighteval:aime24:0       |      2| math_pass@1:1_samples  |0.0667|± |0.0463|
|                          |       | math_pass@1:4_samples  |0.0500|± |0.0279|
|                          |       | math_pass@1:8_samples  |0.0417|± |0.0219|
|                          |       | math_pass@1:16_samples |0.0604|± |0.0269|
|                          |       | math_pass@1:32_samples |0.0646|± |0.0271|
|                          |       | math_pass@1:64_samples |0.0651|± |0.0268|
| lighteval:aime25:0       |      2| math_pass@1:1_samples  |0.0667|± |0.0463|
|                          |       | math_pass@1:4_samples  |0.0750|± |0.0363|
|                          |       | math_pass@1:8_samples  |0.0667|± |0.0310|
|                          |       | math_pass@1:16_samples |0.0667|± |0.0287|
|                          |       | math_pass@1:32_samples |0.0792|± |0.0328|
|                          |       | math_pass@1:64_samples |0.0698|± |0.0308|
| lighteval:gpqa:diamond:0 |      1| gpqa_pass@1:1_samples  |0.3131|± |0.0330|
|                          |       | gpqa_pass@1:4_samples  |0.3548|± |0.0220|
|                          |       | gpqa_pass@1:8_samples  |0.3636|± |0.0193|
| lighteval:gsm8k:0        |      0| extractive_match       |0.4443|± |0.0137|