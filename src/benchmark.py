from utils import (
    get_models_from_user, get_k_values_from_user, get_query_count_from_user,
    TRUE_QUERIES, FALSE_QUERIES, GROUND_TRUTHS, EMBEDDING_MODELS, run_benchmark
)

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

if __name__ == "__main__":
    console = Console()
    console.print(Panel("Custom benchmark run", style="bold magenta"))
    models_to_run = get_models_from_user(available_models=EMBEDDING_MODELS, test_mode=True)
    k_values_to_run = get_k_values_from_user(test_mode=True)

    selected_true_q = get_query_count_from_user(TRUE_QUERIES, "true")
    selected_false_q = get_query_count_from_user(FALSE_QUERIES, "false")
    queries_to_run = selected_true_q + selected_false_q

    if not queries_to_run:
        console.print(Panel("[INFO] No queries selected. Benchmark will stop.", style="bold red"))
        exit()

    param_text = Text()
    param_text.append(f"Models: {models_to_run}\n", style="bold green")
    param_text.append(f"k-values: {k_values_to_run}\n", style="bold green")
    param_text.append(f"Number of queries: {len(queries_to_run)} ({len(selected_true_q)} true, {len(selected_false_q)} false)", style="bold green")
    console.print(Panel(param_text, title="Benchmark Parameters", style="bold yellow"))

    run_benchmark(
        models=models_to_run,
        k_values=k_values_to_run,
        queries=queries_to_run,
        ground_truths=GROUND_TRUTHS
    )
