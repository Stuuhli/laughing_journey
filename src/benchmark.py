from utils import (
    get_models_from_user, get_k_values_from_user, get_query_count_from_user,
    TRUE_QUERIES, FALSE_QUERIES, GROUND_TRUTHS, EMBEDDING_MODELS, run_benchmark
)
from rich.console import Console


if __name__ == "__main__":
    console = Console()
    print("Custom benchmark run")
    models_to_run = get_models_from_user(available_models=EMBEDDING_MODELS, test_mode=True)
    k_values_to_run = get_k_values_from_user(test_mode=True)

    selected_true_q = get_query_count_from_user(TRUE_QUERIES, "true")
    selected_false_q = get_query_count_from_user(FALSE_QUERIES, "false")
    queries_to_run = selected_true_q + selected_false_q

    if not queries_to_run:
        print("[INFO] No queries selected. Benchmark will stop.")
        exit()

    print(f"Models: {models_to_run}\n")
    print(f"k-values: {k_values_to_run}\n")
    print(f"Number of queries: {len(queries_to_run)} ({len(selected_true_q)} true, {len(selected_false_q)} false)")

    run_benchmark(
        models=models_to_run,
        k_values=k_values_to_run,
        queries=queries_to_run,
        ground_truths=GROUND_TRUTHS
    )
