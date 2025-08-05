from utils import RESULTS_DIR
from pathlib import Path
import os
import json
import pandas as pd
from datetime import datetime


def parse_config(path: Path):
    # Wer braucht schon exception hanlding???
    if not path.is_file():
        print(f"[ERROR] No config.json found in in {path}")
        return None  # let it crash, let it craaasshhh

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def get_benchmarks() -> str:
    runs = []
    for run_name in os.listdir(RESULTS_DIR):
        run_path = RESULTS_DIR / run_name

        # Nur Ordner sollen hier liegen
        if run_path.is_dir():
            config = parse_config(run_path / "config.json")
            if config:
                runs.append(config)

    for i, run in enumerate(runs):
        ts = datetime.strptime(run['timestamp'], "%Y-%m-%d_%H-%M").strftime("%d.%m.%Y %H:%M")  # convert to human-readable format
        search_methods = ", ".join([m.split(':')[0] for m in run.get('search_methods', [])])
        k_vals_str = ", ".join(map(str, run.get('k_values', [])))
        description = f"Models: {search_methods} | k-Wert: {k_vals_str}"
        print(f"  [{i + 1}] {ts} | {description}")

    while True:
        try:
            choice = int(input("Select number of run (0 zum abort): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(runs):
                return runs[choice - 1]['path']
            else:
                print("Ungültige Nummer. Bitte erneut versuchen.")
        except ValueError:
            print("Bitte eine gültige Zahl eingeben.")


def analyze_benchmark(benchmark_dir: str):
    csv_file = Path(benchmark_dir) / "benchmark_results.csv"

    print(f"nLese Daten aus {csv_file}")
    df = pd.read_csv(csv_file)
    print(df.describe())

    unique_k_vals = sorted(df['k'].unique())
    unique_embedding_models = sorted(df['embedding_model'].unique())
    unique_queries = sorted(df['query'].unique())
    model_score = {}
    k_score = {}
    for model in unique_embedding_models:
        model_score[model] = 0

        for k in unique_k_vals:
            k_score[(model, k)] = 0

            for query in unique_queries:
                results_row = df[(df['embedding_model'] == model) & (df['k'] == k) & (df['query'] == query)]

                if not results_row.empty:
                    score = str(results_row['hit_rank'].iloc[0])

                    if score == '1':
                        k_score[(model, k)] += 5
                    elif score == '2':
                        k_score[(model, k)] += 3
                    elif score == '3':
                        k_score[(model, k)] += 1

            model_score[model] += k_score[(model, k)]
            print(f"k-Score for Model: {model} and k value: {k} is {k_score[(model, k)]}")


selected_benchmark = get_benchmarks()
analyze_benchmark(benchmark_dir=selected_benchmark)
