from datasets import load_dataset
import os
import glob
import json

if __name__ == '__main__':
    output_dir = 'eval_results'
    model_name = 'FreedomIntelligence/openPangu-Embedded-1B'
    revision = 'main'
    tasks = {
        'gsm8k': 'lighteval|gsm8k|0',
        'math_500': 'lighteval|math_500|0',
        'gpqa_diamond': 'lighteval|gpqa:diamond|0',
        'aime24': 'lighteval|aime24|0',
        'aime25': 'lighteval|aime25|0',
        'mmlu': 'original|mmlu|0',
    }
    for task_name, task in tasks.items():
        # details_dir = f"{output_dir}/{model_name.replace('/', '_')}/{revision}/{task_name}/details/{model_name}"
        results_dir = f"{output_dir}/{model_name.replace('/', '_')}/{revision}/{task_name}/results/{model_name}"

        if not os.path.exists(results_dir):
            print(f'Skipping {task_name}, probably not started yet...')
            continue

        # timestamps = glob.glob(f"{details_dir}/*/")
        # timestamp = sorted(timestamps)[-1].split("/")[-2]
        timestamps = glob.glob(f'{results_dir}/*')
        timestamp = sorted(timestamps)[-1].split('/')[-1].split('results_')[-1].split('.json')[0]
        # print(f"Latest timestamp: {timestamp}")

        # details_path = f'{details_dir}/{timestamp}/details_{task}_{timestamp}.parquet'
        results_path = f'{results_dir}/results_{timestamp}.json'

        # if not os.path.exists(details_path) or not os.path.exists(results_path):
        if not os.path.exists(results_path):
            print(f'Skipping {task_name}, probably not done yet...')
            continue

        # Load the results
        results = json.load(open(results_path))
        if task in results['results']:
            task_results = results['results'][task]
        else:
            task_results = results['results']['all']
        means = {k: v for k, v in task_results.items() if 'stderr' not in k}
        stdrs = {k: v for k, v in task_results.items() if 'stderr' in k}
        print('Task', task)
        for mean_k, stdr_k in zip(means, stdrs):
            mean, stderr = means[mean_k] * 100, stdrs[stdr_k] * 100
            print(f'\t- {mean_k:<30s} {mean:.2f} +/- {stderr:.2f}')

        # # Load the details
        # details = load_dataset("parquet", data_files=details_path, split="train")
        # for detail in details:
        #     print(detail)
