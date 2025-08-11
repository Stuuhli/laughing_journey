import random
import json
import pandas as pd
from utils import RESULTS_DIR


def list_runs():
    """List all benchmark runs stored in the results directory.

    Returns:
        List[dict]: Metadata for each valid run.
    """
    runs = []
    if not RESULTS_DIR.exists():
        return runs
    for p in RESULTS_DIR.iterdir():
        if not p.is_dir():
            continue
        cfg = p / "config.json"
        res = p / "benchmark_results.csv"
        if cfg.exists() and res.exists():
            try:
                meta = json.loads(cfg.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            runs.append({
                "path": p,
                "results_csv": res,
                "config": meta,
                "timestamp": meta.get("timestamp", p.name),
                "use_full_chapters": meta.get("use_full_chapters", None),
                "models": meta.get("search_methods", []),
                "k_values": meta.get("k_values", []),
                "num_queries": meta.get("num_queries", None),
            })
    # Neueste zuerst (nach Ordnername/Timestamp heuristisch)
    runs.sort(key=lambda r: r["timestamp"], reverse=True)
    return runs


def pick_run(runs, title):
    """Prompt the user to select a run from a list.

    Args:
        runs (List[dict]): Available runs.
        title (str): Heading shown before the selection.

    Returns:
        dict: Metadata of the selected run.
    """
    print(f"\n{title}")
    for i, r in enumerate(runs, 1):
        print(f"[{i}] {r['timestamp']}  full chapters={r['use_full_chapters']}  models={r['models']}  k={r['k_values']}  nQ={r['num_queries']}  -> {r['path'].name}")
    while True:
        try:
            idx = int(input("> ")) - 1
            if 0 <= idx < len(runs):
                return runs[idx]
        except Exception:
            pass
        print("[ERROR] Ungültige Auswahl. Bitte Zahl eingeben.")


def add_rr(df: pd.DataFrame) -> pd.DataFrame:
    """Add reciprocal rank and boolean hit columns to a DataFrame.

    Args:
        df (pd.DataFrame): Input data containing `hit_rank` and `hit_at_k`.

    Returns:
        pd.DataFrame: Updated DataFrame with `rr` and normalized `hit_at_k`.
    """
    df = df.copy()
    # Robustheit: hit_rank kann float/NaN kommen → erst int-konvertierbar prüfen

    def _rr(x):
        try:
            ix = int(x)
            return 1.0 / ix if ix > 0 else 0.0
        except Exception:
            return 0.0
    df["rr"] = df["hit_rank"].apply(_rr)
    df["hit_at_k"] = df["hit_at_k"].astype(bool)
    return df


def bootstrap_ci_mean(vals, B=1000, alpha=0.05):
    """Calculate a bootstrap confidence interval for the mean.

    Args:
        vals (Iterable[float]): Sample values.
        B (int, optional): Number of bootstrap samples. Defaults to 1000.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        Tuple[float, float]: Lower and upper confidence bounds.
    """
    vals = list(vals)
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    boots = []
    for _ in range(B):
        samp = [vals[random.randrange(n)] for _ in range(n)]
        boots.append(sum(samp) / n)
    boots.sort()
    lo = boots[int((alpha / 2) * B)]
    hi = boots[int((1 - alpha / 2) * B)]
    return lo, hi


def main():
    """Interactive CLI for comparing two benchmark runs.

    Returns:
        None
    """
    runs = list_runs()
    if len(runs) < 2:
        print("[INFO] Finde weniger als zwei gültige Runs in data/results.")
        return
    runA = pick_run(runs, "Wähle Run A (z. B. Small-to-Big):")
    runB = pick_run(runs, "Wähle Run B (z. B. Top-K only):")

    print(f"\nRun A: {runA['path'].name}  mode={runA['config'].get('retrieval_mode', 'unknown')}")
    print(f"\nRun B: {runB['path'].name}  mode={runB['config'].get('retrieval_mode', 'unknown')}")

    a = add_rr(pd.read_csv(runA["results_csv"]))
    b = add_rr(pd.read_csv(runB["results_csv"]))

    # Merge auf (embedding_model, k, query)
    merged = a.merge(b, on=["embedding_model", "k", "query"], suffixes=("_A", "_B"))
    merged["hit_diff"] = merged["hit_at_k_A"].astype(int) - merged["hit_at_k_B"].astype(int)
    merged["rr_diff"] = merged["rr_A"] - merged["rr_B"]

    rows = []
    for (model, k), sub in merged.groupby(["embedding_model", "k"]):
        n = len(sub)
        hit_mean = sub["hit_diff"].mean()
        rr_mean = sub["rr_diff"].mean()
        hit_ci = bootstrap_ci_mean(sub["hit_diff"], B=2000)
        rr_ci = bootstrap_ci_mean(sub["rr_diff"], B=2000)
        frac_A_better = (sub["hit_diff"] > 0).mean()
        frac_equal = (sub["hit_diff"] == 0).mean()
        rows.append({
            "embedding_model": model,
            "k": k,
            "run_A": runA['config'].get('retrieval_mode', 'unknown'),
            "run_B": runB['config'].get('retrieval_mode', 'unknown'),
            "n_queries": n,
            "delta_hit_mean(A-B)": hit_mean,
            "delta_hit_CI_low": hit_ci[0],
            "delta_hit_CI_high": hit_ci[1],
            "delta_MRR_mean(A-B)": rr_mean,
            "delta_MRR_CI_low": rr_ci[0],
            "delta_MRR_CI_high": rr_ci[1],
            "frac_A_better(hit)": frac_A_better,
            "frac_equal(hit)": frac_equal,
        })

    summary = pd.DataFrame(rows).sort_values(["embedding_model", "k"])
    print("\n--- Vergleichs-Summary ---")
    print(summary.to_string(index=False))

    # speichern neben Run A
    out = runA["path"] / "compare_summary_vs_" / runB["path"].name
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "compare_summary.csv", index=False)
    summary.to_excel(out / "compare_summary.xlsx", index=False)
    print(f"\n[OK] Summary gespeichert unter: {out}")

    agg_rows = []
    for modeA, modeB in [(runA['config'].get('retrieval_mode', 'unknown'),
                          runB['config'].get('retrieval_mode', 'unknown'))]:
        agg = {
            "run_A": modeA,
            "run_B": modeB,
            "n_total_queries": summary["n_queries"].sum(),
            "delta_hit_mean(A-B)": summary["delta_hit_mean(A-B)"].mean(),
            "delta_hit_CI_low": bootstrap_ci_mean(summary["delta_hit_mean(A-B)"], B=2000)[0],
            "delta_hit_CI_high": bootstrap_ci_mean(summary["delta_hit_mean(A-B)"], B=2000)[1],
            "delta_MRR_mean(A-B)": summary["delta_MRR_mean(A-B)"].mean(),
            "delta_MRR_CI_low": bootstrap_ci_mean(summary["delta_MRR_mean(A-B)"], B=2000)[0],
            "delta_MRR_CI_high": bootstrap_ci_mean(summary["delta_MRR_mean(A-B)"], B=2000)[1],
            "frac_A_better(hit)": (summary["delta_hit_mean(A-B)"] > 0).mean(),
            "frac_equal(hit)": (summary["delta_hit_mean(A-B)"] == 0).mean(),
        }
        agg_rows.append(agg)

    agg_df = pd.DataFrame(agg_rows)
    print("\n--- Gesamt-Aggregation ---")
    print(agg_df.to_string(index=False))

    # Auch abspeichern
    agg_df.to_csv(out / "compare_summary__AGGREGATED.csv", index=False)
    agg_df.to_excel(out / "compare_summary__AGGREGATED.xlsx", index=False)


if __name__ == "__main__":
    main()

# Metrik-Definitionen:
# delta_hit_mean(A-B): Ø(Hit@k_A - Hit@k_B), Hit=1 falls Ground Truth im Top-k, sonst 0
# delta_hit_CI_low/high: 95%-Bootstrap-Konfidenzintervall für delta_hit_mean
# delta_MRR_mean(A-B): Ø(MRR_A - MRR_B), MRR = 1/(Rank erster Treffer), 0 falls kein Treffer
# delta_MRR_CI_low/high: 95%-Bootstrap-Konfidenzintervall für delta_MRR_mean
# frac_A_better(hit): Anteil Queries mit Hit_A=1 und Hit_B=0
# frac_equal(hit): Anteil Queries mit identischem Hit-Status in A und B
