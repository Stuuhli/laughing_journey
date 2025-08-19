from typing import List, Dict, Any, Optional, Tuple, Iterable
import json
from pathlib import Path
import pandas as pd
import uuid
import datetime
import time
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import random
import os

# --- Konstanten bleiben unverändert ---
PROJECT_DIR = Path(__file__).parent.parent
DOCX_DIR = PROJECT_DIR / "data" / "docx"
CONVERTED_DIR = PROJECT_DIR / "data" / "converted"
CHUNKS_DIR = PROJECT_DIR / "data" / "chunks"
RESULTS_DIR = PROJECT_DIR / "data" / "results"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma"


PROMPT_TEMPLATE = """
Du bist ein präziser und hilfreicher Assistent in einem RAG-System.
Antworte ausschließlich auf Basis des folgenden Kontexts.

Jeder Kontextblock beginnt mit einer Quellnummer in eckigen Klammern, z. B. [1], [2].
Verweise in deiner Antwort **direkt im Fließtext** an den relevanten Stellen auf diese Quellnummern
(z. B. „… gemäß [1] …“, „… siehe [2][3] …“). Nutze nur die vorhandenen Quellnummern, erfinde keine.

Wenn die Information im Kontext fehlt, sage das explizit.

Deine Antworten sollen im Markdown-Format sein.
Wiederhole niemals denselben Satz oder dieselbe Quelle mehrfach. Nutze jede Quelle höchstens einmal.

Kontext:
{context}

Frage:
{query}

Antwort (mit Quellenangaben im Text):
"""

# Spezieller Prompt für den Vergleichsmodus (Inkonsistenzen etc.)
PROMPT_TEMPLATE_COMPARE = """
Du vergleichst mehrere vollständige Dokumente inhaltlich. Der gesamte Kontext folgt in nummerierten Blöcken [1], [2], ...

Aufgaben:
1) Beantworte die Frage **präzise** ausschließlich auf Basis des Kontexts.
2) Führe eine **Konsistenzprüfung** durch: identifiziere Widersprüche, fehlende Anforderungen, divergierende Definitionen, Versionskonflikte, unterschiedliche Verantwortlichkeiten.
3) Verweise im Fließtext stets auf die Quellnummern [n] an den relevanten Stellen.
4) Wenn Angaben fehlen oder zwischen Quellen variieren, formuliere **konkrete Klärungsfragen**.

Kontext (nummerierte Quellenblöcke):
{context}

Frage:
{query}

Antwort (mit Quellenangaben im Text):
"""


TRUE_QUERIES = [
    "Was sind Ziele der Informationssicherheit?",  # Leit
    "Was ist Business Continuity?",
    "Wer hat die Leitlinie unterschrieben?",
    "Was sind mobile Datenträger?",  # Richt
    "Kann ich die Tür in mein Büro in der Pause offen lassen?",
    "Wie muss mein Smartphone entsperrt werden?",
    "Welche Anforderungen sind explizit an Passwörter gestellt?",  # Pass
    "Wie sind Passwörter aufzubewahren?",
    "Was zählt als Sicherheitsvorfall?",
    "Wie sind vertrauliche Papierdokumente zu vernichten?",  # ZZZ
    "Welche Sicherheitszonen gibt es?",
    "Nach welchen Normen muss vernichtet werden und was sagen diese aus?",  # 12
    "Who is the CISO?",  # Guide
    "How should I act in case of a suspicious applicaiton",
    "What do I have to consider when Teleworking",
    "Whats's the scope of the policy?",  # Policy
    "What roles exist in the organization?",
    "What can you tell me about data protection and privacy?",
]
FALSE_QUERIES = [
    "Bis wann mus ich meine Steuererklärung abgeben?",
    "Wie laut brummt eine Hummel?",
    "Wenn Peter doppelt so viel wiegt wie Anna und Anna wiegt 40kg wie schwer sind beide zusammen?",
    "Liebst du mich?",
    "Was ist der Sinn des Lebens?",
    "Welche Vorgaben gibt es für Bürostühle?",
    "Im Falle eines Feuers wie ist mit Sicherheitstüren umzugehen?",
    "Wenn ich mein Smartphone mit sensiblen Daten und Firmengeheimnissen drauf verliere was muss ich tun?",
    "Wie lauten die Namen der Vorstandsmitglieder?",
    "Was ist der firmenweite kryptographische Mindeststandard?",
]
GROUND_TRUTHS = {
    "Was sind Ziele der Informationssicherheit?": "Das Ziel der Informationssicherheit bei KISTERS",
    "Was ist Business Continuity?": "Im Falle eines Notfalls oder einer Krise",
    "Wer hat die Leitlinie unterschrieben?": "Klaus Kisters, Vorstand",
    "Was sind mobile Datenträger?": "[INFO] Mobile Datenträger sind u.a.",
    "Kann ich die Tür in mein Büro in der Pause offen lassen?": "Räume mit hohem Schutzbedarf MÜSSEN",
    "Wie muss mein Smartphone entsperrt werden?": "Für Smartphones gelten zusätzlich folgende Maßnahmen",
    "Welche Anforderungen sind explizit an Passwörter gestellt?": "[INFO] Passwörter sollen „schwer zu knacken, aber leicht zu behalten“ sein",
    "Wie sind Passwörter aufzubewahren?": "Für den Umgang mit Passwörtern sind die folgenden besonderen Regeln",
    "Was zählt als Sicherheitsvorfall?": "Wenn ein Mitarbeiter die Kompromittierung eines Passwortes",
    "Wie sind vertrauliche Papierdokumente zu vernichten?": "Papierdokumente MÜSSEN mit Akten",
    "Welche Sicherheitszonen gibt es?": "Für die Zutrittsregelung von Betriebsstätten",
    "Nach welchen Normen muss vernichtet werden und was sagen diese aus?": "Die Vernichtung von Datenträgern MUSS auf Basis der Klassifikation",
    "Who is the CISO?": "Dr. Heinz-Josef Schlebusch",
    "How should I act in case of a suspicious applicaiton": "In the event that a computer or application",
    "What do I have to consider when Teleworking": "Teleworking sites are workplaces that are",
    "Whats's the scope of the policy?": "This policy defines the KISTERS general strategic",
    "What roles exist in the organization?": "The overall responsibility for information security",
    "What can you tell me about data protection and privacy?": "The protection of natural persons in relation",
}

EMBEDDING_MODELS = [
    "all-minilm:33m",
    "bge-large:335m",
    "bge-m3:567m",
    "granite-embedding:30m",
    "granite-embedding:278m",
    "mxbai-embed-large:335m",
    "nomic-embed-text:v1.5",
    "snowflake-arctic-embed2:568m",
]
EMBEDDING_MAX_LENGTHS = {
    "bge-m3:567m": 8192,
    "snowflake-arctic-embed2:568m": 8192,
    "nomic-embed-text:v1.5": 2048,
    "mxbai-embed-large:335m": 512,
    "all-minilm:33m": 512,
    "bge-large:335m": 512,
    "granite-embedding:278m": 512,  # der King, the GOAT
    "granite-embedding:30m": 512,
}
# TODO: Aktuelle größte Tokenlength etwa 1.000 - 1.300 bei 5182 Zeichen page_context

GENERATION_MODELS = [
    "deepseek-r1:1.5b",
    "gemma3n:e2b",
    "phi3:3.8b",
    "gemma3n:e4b",
    "qwen:4b",
    "granite3.3:8b",
    "llama3.1:8b",
    "qwen3:8b",
    "gemma3:12b",
    "mistral-nemo:12b",
]

"""
Model               context length
mistral-nemo:12b:   1024000
gemma3:12b:         131072
granite3.3:8b:      131072
llama3.1:8b:        131072
phi3:3.8b:          131072
deepseek-r1:1.5b:   131072
qwen3:8b:           40960
gemma3n:e4b:        32768
qwen:4b:            32768
gemma3n:e2b:        32768
mistral:latest:     32768
"""


def is_relevant_chunk(chunk_text, ground_truth):
    """Determine if a chunk contains the expected ground truth string.

    Args:
        chunk_text (str): Text of the chunk being inspected.
        ground_truth (str): String that should appear within the chunk.

    Returns:
        bool: True if the ground truth is found in the chunk.
    """
    return ground_truth.lower() in chunk_text.lower()


# --- NEUE GENERISCHE FUNKTION ---
def generic_let_user_choose(
    prompt: str,
    options: Optional[List[Any]] = None,
    allow_multiple: bool = False,
    validate_func: callable = None
) -> List[Any]:
    """
    A generic function for querying user input.

    Args:
        prompt (str): The prompt for the user like "Please choose value k"
        options (Optional[List[Any]]): A list of predefined options.
        allow_multiple (bool): Allows multiple options to be selected (separated by commas).
        validate_func (callable): An optional function for validating the input
                                  (e.g. for number ranges).

    Returns:
        List[Any]: A list of valid user selections.
    """
    print(f"\n{prompt}")

    if options:
        for idx, option in enumerate(options):
            print(f"  [{idx + 1}] {option}")

    while True:
        user_input = input("> ").strip()
        if not user_input:
            print("[ERROR] No input received. Please select at least one option.")
            continue

        parts = [p.strip() for p in user_input.split(',')]
        if not allow_multiple and len(parts) > 1:
            print("[ERROR] Multiple options selected, but only one is allowed.")
            continue

        selection = []
        is_valid = True

        for part in parts:
            if validate_func:
                if not validate_func(part):
                    is_valid = False
                    break
                selection.append(part)
            elif options:
                try:
                    choice_idx = int(part) - 1
                    if 0 <= choice_idx < len(options):
                        selection.append(options[choice_idx])
                    else:
                        print(f"[ERROR] Number '{part}' is invalid. Please use numbers between 1 and {len(options)}.")
                        is_valid = False
                        break
                except ValueError:
                    print(f"[ERROR] Invalid input: '{part}'. Please use numbers only.")
                    is_valid = False
                    break
            else:  # Freitext oder einfache Ja/Nein-Logik
                selection.append(user_input)
                break

        if is_valid and selection:
            if not options and not allow_multiple:
                return selection

            result = sorted(list(set(selection)), key=selection.index)
            if not allow_multiple:
                return [result[0]]
            return result
        elif is_valid and not selection:
            print("[ERROR] Invalid selection. Please try agin.")


def get_models_from_user(available_models: List[str], test_mode: bool = False) -> List[str]:
    """Prompt the user to select one or more embedding models.

    Args:
        available_models (List[str]): Models the user can choose from.
        test_mode (bool, optional): Allow multiple selections when True. Defaults to False.

    Returns:
        List[str]: The chosen model names.
    """
    return generic_let_user_choose(
        prompt="1. Select model(s) by number (e.g. 1,3):" if test_mode else "1. Choose one model by number (e.g. 1):",
        options=available_models,
        allow_multiple=test_mode
    )


def get_k_values_from_user(test_mode: bool = False) -> List[int]:
    """Ask the user for one or more ``k`` values.

    Args:
        test_mode (bool, optional): Allow multiple selections when True. Defaults to False.

    Returns:
        List[int]: The selected ``k`` values.
    """

    # Validation function for k values
    def validate_k_value(val: str) -> bool:
        if val.isdigit() and int(val) > 0:
            return True
        print(f"[ERROR] invalid input: '{val}'. Please only use positive, natural numbers.")
        return False

    selected_values = generic_let_user_choose(
        prompt="2. Which k values should be tested? (e.g. 2,4,6)" if test_mode else "2. Which k value should be used? (e.g. 3)",
        allow_multiple=test_mode,
        validate_func=validate_k_value
    )
    return [int(val) for val in selected_values]


def get_query_count_from_user(query_list: List[str], query_type_name: str) -> List[str]:
    """Select how many queries of a certain type to use.

    Args:
        query_list (List[str]): Available queries.
        query_type_name (str): Label for the query type (e.g., ``"true"`` or ``"false"``).

    Returns:
        List[str]: A subset of ``query_list`` with the desired length.
    """
    max_count = len(query_list)
    prompt = (
        f"3.1 How many '{query_type_name}' queries should be tested? (max: {max_count})"
        if query_type_name == 'true'
        else f"3.2 How many '{query_type_name}' queries should be tested? (max: {max_count})"
    )

    # Validation function for the number of queries
    def validate_query_count(val: str) -> bool:
        try:
            count = int(val)
            if 0 <= count <= max_count:
                return True
            print(f"[ERROR] Invalid input. Please enter a number between 0 and {max_count}.")
            return False
        except ValueError:
            print("[ERROR] Invalid input. Please enter a whole natural number.")
            return False

    count_str = generic_let_user_choose(
        prompt=prompt, allow_multiple=False, validate_func=validate_query_count
    )[0]
    count = int(count_str)
    return query_list[:count]


def get_search_method_from_user():
    """Ask the user which search method to use.

    Returns:
        str: ``"normal"`` or ``"hyde"`` depending on the selection.
    """
    options = ["Normal search (original query)", "HyDE-search (hypothetical document embedding)"]
    choice = generic_let_user_choose(
        prompt="Which search method should be used?",
        options=options,
        allow_multiple=False
    )
    return "normal" if choice[0] == options[0] else "hyde"


def load_chunks(chunk_dir=None) -> List[Dict[str, Any]]:
    """Load chunk data from JSON files in a directory.

    Args:
        chunk_dir (Path, optional): Directory containing chunk files. Defaults to ``CHUNKS_DIR``.

    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries.
    """
    if chunk_dir is None:
        chunk_dir = CHUNKS_DIR
    chunk_files = [f for f in Path(chunk_dir).glob("*.json")]
    chunks = []
    max_length = 0
    max_length_file = ""
    max_length_chunk_id = ""
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            file_chunks = json.load(f)
            chunks.extend(file_chunks)
            for chunk in file_chunks:
                current_length = len(chunk['page_content'])
                if current_length > max_length:
                    max_length = current_length
                    max_length_file = chunk['metadata']['source_file']
                    max_length_chunk_id = chunk['metadata']['chunk_id']
    if chunks:
        print(
            f"\n[INFO] Maximum chunk size (character): {max_length} for file {max_length_file} and chunk id {max_length_chunk_id}"
        )
    return chunks


def get_embedding_object(model_name: str):
    """Instantiate an embedding object for a given model name.

    Args:
        model_name (str): Name of the embedding model.

    Returns:
        OllamaEmbeddings: Embedding model instance.
    """
    num_ctx = EMBEDDING_MAX_LENGTHS[model_name]
    return OllamaEmbeddings(model=model_name, num_ctx=num_ctx)


def get_context_mode_from_user() -> bool:
    """Choose between the small-to-big and top-k context strategies.

    Returns:
        bool: True for small-to-big mode, False for top-k only.
    """
    options = [
        "Small-to-Big (volle Kapitel aus Treffern laden)",
        "Nur Top-K direkte Treffer verwenden"
    ]
    choice = generic_let_user_choose(
        prompt="Which context strategy should be used?",
        options=options,
        allow_multiple=False
    )
    return choice[0] == options[0]


def run_benchmark(models: List[str], k_values: List[int], queries: List[str], ground_truths: Dict[str, str], use_full_chapters: bool = True):
    """Run the retrieval benchmark for a set of models and queries.

    Args:
        models (List[str]): Embedding models to evaluate.
        k_values (List[int]): Values of ``k`` to test.
        queries (List[str]): Queries to execute.
        ground_truths (Dict[str, str]): Mapping of queries to expected text snippets.
        use_full_chapters (bool, optional): Use chapter context instead of top-k chunks. Defaults to True.

    Returns:
        None
    """
    chunks = load_chunks()
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    results = []

    total_tasks = len(models) * len(k_values) * len(queries)
    task_count = 0
    finished_models = []
    overall_start = time.time()

    print("[INFO] Running benchmark...")
    for model in models:
        model_start = time.time()
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MODEL] Starting: {model}")

        chunk_dir = get_chunk_dir_for_model(model)
        chunks = load_chunks(chunk_dir=chunk_dir)
        docs = [Document(page_content=c["page_content"], metadata=c.get("metadata", {})) for c in chunks]

        emb = get_embedding_object(model)
        col_name = f"benchmark_collection_{uuid.uuid4().hex}"
        db = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=None, collection_name=col_name)

        for k in k_values:
            for query in queries:
                res = db.similarity_search_with_score(query, k=k)
                ground_truth = ground_truths.get(query)
                hit_found, hit_rank, score, doc_title, content = False, -1, None, None, None

                if use_full_chapters:
                    # Retrieve all chunks belonging to the chapter of the first hit
                    if res:
                        first_doc = res[0][0]
                        chapter_id = first_doc.metadata.get("chapter_path")
                        if chapter_id:
                            chapter_chunks = [
                                c["page_content"]
                                for c in chunks
                                if c["metadata"].get("chapter_path") == chapter_id
                            ]
                            content = "\n".join(chapter_chunks)
                        else:
                            content = first_doc.page_content
                        doc_title = first_doc.metadata.get("source_file", "N/A")

                        if ground_truth and any(is_relevant_chunk(c, ground_truth) for c in chapter_chunks):
                            hit_found = True
                            hit_rank = 1
                            score = res[0][1]

                else:
                    # Normal mode: check only the returned top-k chunks
                    for idx, (doc, score_val) in enumerate(res, 1):
                        if ground_truth and is_relevant_chunk(doc.page_content, ground_truth):
                            hit_found = True
                            hit_rank = idx
                            doc_title = doc.metadata.get('source_file', 'N/A')
                            content = doc.page_content
                            score = score_val
                            break

                if not hit_found:
                    if res:
                        doc, score_val = res[0]
                        doc_title = doc.metadata.get('source_file', 'N/A')
                        content = content or doc.page_content
                        score = score_val
                    else:
                        doc_title = "N/A"
                        content = "Ground truth could not be found"
                        score = None

                results.append({
                    'embedding_model': model,
                    'k': k,
                    'query': query,
                    'query_type': 'true' if query in TRUE_QUERIES else 'false',
                    'score': score,
                    'hit_at_k': hit_found,
                    'hit_rank': hit_rank,
                    'titel': doc_title,
                    'content': content,
                })
                task_count += 1
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {task_count}/{total_tasks} {model} k={k} query='{query}' complete")

        model_time = time.time() - model_start
        finished_models.append((model, model_time))
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MODEL] Finished: {model} in {model_time:.2f} seconds")
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MODEL] Models completed so far: {[m[0] for m in finished_models]}")

    overall_time = time.time() - overall_start
    df = pd.DataFrame(results)
    summary_rows = []
    for (model, k), sub in df.groupby(["embedding_model", "k"]):
        rows = sub[["hit_at_k", "hit_rank"]].to_dict(orient="records")
        base = compute_metrics_from_rows(rows)
        boot = bootstrap_metrics(rows, B=500)
        summary_rows.append({
            "embedding_model": model,
            "k": k,
            "use_full_chapters": use_full_chapters,
            "queries": base["queries"],
            "hit_rate": base["hit_rate"],
            "hit_rate_ci_low": boot["hit_ci"][0],
            "hit_rate_ci_high": boot["hit_ci"][1],
            "mrr": base["mrr"],
            "mrr_ci_low": boot["mrr_ci"][0],
            "mrr_ci_high": boot["mrr_ci"][1],
        })
    results_subdir = (
        RESULTS_DIR / f"{timestamp}__{len(models)}-models"
        if len(models) != len(EMBEDDING_MODELS)
        else RESULTS_DIR / f"{timestamp}-all-models"
    )
    results_subdir.mkdir(exist_ok=True, parents=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_subdir / "benchmark_summary.csv", index=False)
    summary.to_excel(results_subdir / "benchmark_summary.xlsx", index=False)
    df.to_csv(results_subdir / "benchmark_results.csv", index=False)
    df.to_excel(results_subdir / "benchmark_results.xlsx", index=False)

    run_config = {
        "timestamp": timestamp,
        "path": str(results_subdir),
        "search_methods": models,
        "k_values": k_values,
        "num_queries": len(queries),
        "use_full_chapters": use_full_chapters,
        "retrieval_mode": "small_to_big" if use_full_chapters else "topk_only",
    }
    with open(results_subdir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=4)

    print(f"\n[SUCCESS] Benchmark completed. Results are in: {results_subdir}")
    print(f"[TIME] Total benchmark time: {overall_time:.2f} seconds")
    for m, t in finished_models:
        print(f"[TIME] {m}: {t:.2f} seconds")


def sliding_window_chunk(text, max_length, stride=0.2):
    """Split text into overlapping chunks using a sliding window.

    Args:
        text (str): The input text.
        max_length (int): Maximum length of each chunk.
        stride (float, optional): Overlap ratio between chunks. Defaults to 0.2.

    Returns:
        list: Generated text chunks.
    """
    window = max_length
    step = int(window * (1 - stride))
    return [text[i:i + window] for i in range(0, max(len(text) - window + 1, 1), step)]


def get_chunk_dir_for_model(model_name: str):
    """Return the directory path for chunks of a specific model.

    Args:
        model_name (str): Embedding model name.

    Returns:
        Path: Path to the model's chunk directory.
    """
    safe_name = model_name.replace(":", "-").replace("/", "-")
    return Path(__file__).parent.parent / "data" / "chunks" / f"chunks__{safe_name}"


def _reciprocal_rank(rank: int) -> float:
    """Calculate the reciprocal rank for a given position.

    Args:
        rank (int): Rank of the first relevant document.

    Returns:
        float: Reciprocal rank value.
    """
    return 1.0 / rank if rank > 0 else 0.0


def compute_metrics_from_rows(rows):
    """Compute hit rate and mean reciprocal rank from result rows.

    Args:
        rows (List[dict]): Rows containing ``hit_at_k`` and ``hit_rank``.

    Returns:
        dict: Metrics including query count, hit rate and MRR.
    """
    n = len(rows) or 1
    hit = sum(1 for r in rows if r.get("hit_at_k"))
    mrr = sum(_reciprocal_rank(r.get("hit_rank", -1)) for r in rows) / n
    return {
        "queries": n,
        "hit_rate": hit / n,
        "mrr": mrr,
    }


def bootstrap_metrics(rows, B=500):
    """Bootstrap hit rate and MRR metrics from benchmark rows.

    Args:
        rows (List[dict]): Result rows with ``hit_at_k`` and ``hit_rank``.
        B (int, optional): Number of bootstrap samples. Defaults to 500.

    Returns:
        dict: Mean metrics and confidence intervals.
    """
    n = len(rows)
    samples = []
    for _ in range(B):
        samp = [rows[random.randrange(n)] for _ in range(n)]
        samples.append(compute_metrics_from_rows(samp))

    def ci(vals):
        vals = sorted(vals)
        lo = vals[int(0.025 * B)]
        hi = vals[int(0.975 * B)]
        return (lo, hi)

    hit_vals = [s["hit_rate"] for s in samples]
    mrr_vals = [s["mrr"] for s in samples]
    return {
        "hit_mean": sum(hit_vals) / B,
        "hit_ci": ci(hit_vals),
        "mrr_mean": sum(mrr_vals) / B,
        "mrr_ci": ci(mrr_vals),
    }


# =========================
#   Vergleichen: Helpers
# =========================

def list_docx_files() -> List[Path]:
    """List available DOCX files.

    Returns:
        List[Path]: Sorted list of document paths.
    """
    DOCX_DIR.mkdir(exist_ok=True, parents=True)
    return sorted(DOCX_DIR.glob("*.docx"))


def ensure_converted_text_for_docx(docx_path: Path) -> Path:
    """Ensure a converted text file exists for the given DOCX document.

    Args:
        docx_path (Path): Path to the source DOCX file.

    Returns:
        Path: Path to the corresponding text file.
    """
    CONVERTED_DIR.mkdir(exist_ok=True, parents=True)
    txt_name = Path(os.path.splitext(docx_path.name)[0] + ".txt")
    out_path = CONVERTED_DIR / txt_name
    if out_path.exists():
        return out_path

    # Lazy import to avoid circular dependency
    from docling_converter import convert_all_docx

    print(f"[INFO] Converted text not found for '{docx_path.name}', converting via docling...")
    convert_all_docx(update=False, doc_path=docx_path)
    if not out_path.exists():
        raise FileNotFoundError(f"[ERROR] Docling conversion did not produce: {out_path}")
    return out_path


def load_chunks_for_source(embedding_model_name: str, source_txt_filename: str) -> List[str]:
    """Load all chunk texts for a specific source file and model.

    Args:
        embedding_model_name (str): Embedding model name.
        source_txt_filename (str): Name of the source text file.

    Returns:
        List[str]: List of chunk contents.
    """
    chunk_dir = get_chunk_dir_for_model(embedding_model_name)
    if not chunk_dir.exists():
        return []

    contents: List[str] = []
    for jf in chunk_dir.glob("*.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                items = json.load(f)
            for it in items:
                meta = it.get("metadata", {})
                if meta.get("source_file") == source_txt_filename:
                    contents.append(it.get("page_content", ""))
        except Exception as e:
            print(f"[WARN] Could not read chunks from {jf}: {e}")
            continue
    return contents


def build_compare_context(selected_docx: List[Path], embedding_model_name: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Build a numbered context string over multiple sources.

    Preference is given to existing chunks for the given model; otherwise the
    converted full text is used.

    Args:
        selected_docx (List[Path]): Documents to include.
        embedding_model_name (str): Model used to locate chunks.

    Returns:
        Tuple[str, List[Tuple[int, str]]]: Context string and list of source labels.
    """
    parts: List[str] = []
    sources: List[Tuple[int, str]] = []

    for i, docx in enumerate(selected_docx, start=1):
        source_txt_filename = Path(os.path.splitext(docx.name)[0] + ".txt").name

        # Try chunks first
        chunk_texts = load_chunks_for_source(embedding_model_name, source_txt_filename)

        if chunk_texts:
            combined = "\n".join(chunk_texts)
            parts.append(f"[{i}] {combined}")
            sources.append((i, f"{source_txt_filename} (Chunks)"))
            continue

        # Fallback: converted full text
        txt_path = ensure_converted_text_for_docx(docx)
        text_content = txt_path.read_text(encoding="utf-8")
        parts.append(f"[{i}] {text_content}")
        sources.append((i, f"{source_txt_filename} (Converted TXT)"))

    context_string = "\n\n--- DOC SEP ---\n\n".join(parts) if parts else ""
    return context_string, sources


def stream_to_markdown(stream: Iterable[str], placeholder) -> str:
    """Render a token stream progressively inside a Streamlit placeholder.
    This utility consumes the incoming text chunks, updates the given
    placeholder with the accumulated markdown and returns the final string.
    Keeping the logic in one place avoids subtle differences in how streaming
    is handled across the code base.
    Args:
        stream: Iterable yielding pieces of text from an LLM.
        placeholder: Streamlit placeholder created via ``st.empty()``.
    Returns:
        str: The concatenated response string.
    """

    response = ""
    for chunk in stream:
        response += chunk
        placeholder.markdown(response)
    return response
