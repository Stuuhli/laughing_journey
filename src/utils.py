from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import pandas as pd
import uuid
import datetime
import time
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

# --- Konstanten bleiben unverändert ---
PROJECT_DIR = Path(__file__).parent.parent
DOCX_DIR = PROJECT_DIR / "data" / "docx"
CONVERTED_DIR = PROJECT_DIR / "data" / "converted"
CHUNKS_DIR = PROJECT_DIR / "data" / "chunks"
RESULTS_DIR = PROJECT_DIR / "data" / "results"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma"

PROMPT_TEMPLATE = """
Du bist ein präziser und hilfreicher Assistent in einem RAG-System.
Antworte auf die Frage des Benutzers ausschließlich auf Basis des folgendes Kontexts.
Jeder Kontextabschnitt beinhaltet auch Metadaten, auf die du dich beziehen sollst.
Die Schlüsselwörter (MUSS, DARF NUR, VERPFLICHTEND), (SOLL, EMPFOHLEN), (KANN, OPTIONAL)
MÜSSEN von dir in den Antworten ebenfalls groß geschrieben werden, und MÜSEEN mit *fett* gemacht werden.

Deine Aufgaben:
1.  Formuliere eine präzise Antwort auf die Frage.
2.  Wenn die benötigten Fakten für deine Antwort im Kontext nicht enthalten ist, sage klar und deutlich,
    dass du keine Antwort finden konntest. Erfinde nichts. Bleibe immer bei dem Kontext.
3.  Erwähne stets, dass sich deine Informationen nur auf den gefundenen Kontext,
    bzw. die Quellen beziehen und weitere Informationen fehlen könnten.
4.  Zitiere am Ende deiner Antwort für JEDE verwendete Information
    IMMER die genaue Quelle und die Seitenzahl im Format [Dokument-Quelle, Seite X].
    Du kannst mehrere Quellen zitieren.

Kontext:
{context}

Frage:
{query}

Antwort:
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
qwen3:8b:           40960
gemma3n:e4b:        32768
qwen:4b:            32768
gemma3n:e2b:        32768
mistral:latest:     32768
"""


def is_relevant_chunk(chunk_text, ground_truth):
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
    return generic_let_user_choose(
        prompt="1. Select model(s) by number (e.g. 1,3):" if test_mode else "1. Choose one model by number (e.g. 1):",
        options=available_models,
        allow_multiple=test_mode
    )


def get_k_values_from_user(test_mode: bool = False) -> List[int]:
    # Validierungsfunktion für k-Werte
    def validate_k_value(val: str) -> bool:
        if val.isdigit() and int(val) > 0:
            return True
        print(f"[ERROR] invalid input: '{val}'. Please only use positive, natural numbers.")
        return False

    selected_values = generic_let_user_choose(
        prompt="2. Which k values should be tested? (e.g. 2,4,6)" if test_mode else "2. Which k value should be tested? (e.g. 3)",
        allow_multiple=test_mode,
        validate_func=validate_k_value
    )
    return [int(val) for val in selected_values]


def get_query_count_from_user(query_list: List[str], query_type_name: str) -> List[str]:
    max_count = len(query_list)
    prompt = f"3.1 How many '{query_type_name}' queries should be tested? (max: {max_count})" if query_type_name == 'true' else f"3.2 How many '{query_type_name}' queries should be tested? (max: {max_count})"

    # Validierungsfunktion für die Anzahl der Queries
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

    count_str = generic_let_user_choose(prompt=prompt, allow_multiple=False, validate_func=validate_query_count)[0]
    count = int(count_str)
    return query_list[:count]


def get_search_method_from_user():
    options = ["Normal search (original query)", "HyDE-search (hypothetical document embedding)"]
    choice = generic_let_user_choose(
        prompt="Which search method should be used?",
        options=options,
        allow_multiple=False
    )
    return "normal" if choice[0] == options[0] else "hyde"


def load_chunks(chunk_dir=None) -> List[Dict[str, Any]]:
    if chunk_dir is None:
        chunk_dir = CHUNKS_DIR
    chunk_files = [f for f in Path(chunk_dir).glob("*.json")]
    chunks = []
    max_length = 0
    max_length_file = ""
    max_length_chunk_id = ""
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            # chunks.extend(json.load(f))
            file_chunks = json.load(f)
            chunks.extend(file_chunks)
            for chunk in file_chunks:
                current_length = len(chunk['page_content'])
                if current_length > max_length:
                    max_length = current_length
                    max_length_file = chunk['metadata']['source_file']
                    max_length_chunk_id = chunk['metadata']['chunk_id']
    if chunks:
        print(f"\n[INFO] Maximale Chunk-Größe (Zeichen): {max_length} for file {max_length_file} and chunk id {max_length_chunk_id}")
    return chunks


def get_embedding_object(model_name: str):
    num_ctx = EMBEDDING_MAX_LENGTHS[model_name]
    return OllamaEmbeddings(model=model_name, num_ctx=num_ctx)


def run_benchmark(models: List[str], k_values: List[int], queries: List[str], ground_truths: Dict[str, str]):
    chunks = load_chunks()

    docs = [Document(page_content=c["page_content"], metadata=c.get("metadata", {})) for c in chunks]
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    results = []

    for model in models:
        chunk_dir = get_chunk_dir_for_model(model)
        chunks = load_chunks(chunk_dir=chunk_dir)
        docs = [Document(page_content=c["page_content"], metadata=c.get("metadata", {})) for c in chunks]

    total_tasks = len(models) * len(k_values) * len(queries)
    task_count = 0
    finished_models = []
    overall_start = time.time()

    print("[INFO] Running benchmark...")
    for model in models:
        model_start = time.time()
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MODEL] Starting: {model}")
        emb = get_embedding_object(model)
        col_name = f"benchmark_collection_{uuid.uuid4().hex}"
        db = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=None, collection_name=col_name)  # benchmark-db ist nicht persistent
        for k in k_values:
            for query in queries:
                res = db.similarity_search_with_score(query, k=k)
                ground_truth = ground_truths.get(query)
                hit_found, hit_rank, score, doc_title, content = False, -1, None, None, None

                for idx, (doc, score_val) in enumerate(res, 1):
                    if ground_truth and is_relevant_chunk(doc.page_content, ground_truth):
                        hit_found = True
                        hit_rank = idx
                        doc_title = doc.metadata.get('source_file', 'N/A')
                        content = doc.page_content
                        score = score_val
                        break

                if not hit_found:
                    content = "Ground truth could not be retrieved from chunks"
                    if res:
                        doc, score_val = res[0]
                        doc_title = doc.metadata.get('source_file', 'N/A')
                        score = score_val
                    else:
                        doc_title = "N/A"
                        score = None

                results.append({
                    'embedding_model': model, 'k': k, 'query': query,
                    'query_type': 'true' if query in TRUE_QUERIES else 'false',
                    'score': score, 'hit_at_k': hit_found, 'hit_rank': hit_rank,
                    'titel': doc_title, 'content': content,
                })
                task_count += 1
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {task_count}/{total_tasks} {model} k={k} query='{query}' complete")

        model_time = time.time() - model_start
        finished_models.append((model, model_time))
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MODEL] Finished: {model} in {model_time:.2f} seconds")
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [MODEL] Models completed so far: {[m[0] for m in finished_models]}")

    overall_time = time.time() - overall_start
    df = pd.DataFrame(results)
    if len(models) != len(EMBEDDING_MODELS):
        results_subdir = RESULTS_DIR / f"{timestamp}__{len(models)}-models"
    else:
        results_subdir = RESULTS_DIR / f"{timestamp}-all-models"
    results_subdir.mkdir(exist_ok=True, parents=True)
    df.to_csv(results_subdir / "benchmark_results.csv", index=False)
    df.to_excel(results_subdir / "benchmark_results.xlsx", index=False)

    run_config = {
        "timestamp": timestamp,
        "path": str(results_subdir),
        "search_methods": models,
        "k_values": k_values,
        "num_queries": len(queries)
    }
    with open(RESULTS_DIR / results_subdir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=4)

    print(f"\n[SUCCESS] Benchmark completed. Results are in: {results_subdir}")
    print(f"[TIME] Total benchmark time: {overall_time:.2f} seconds")
    for m, t in finished_models:
        print(f"[TIME] {m}: {t:.2f} seconds")


def sliding_window_chunk(text, max_length, stride=0.2):
    """Teile Text mit Sliding Window in überlappende Chunks."""
    window = max_length
    step = int(window * (1 - stride))
    return [text[i:i + window] for i in range(0, max(len(text) - window + 1, 1), step)]


def get_chunk_dir_for_model(model_name: str):
    safe_name = model_name.replace(":", "-").replace("/", "-")
    return Path(__file__).parent.parent / "data" / "chunks" / f"chunks__{safe_name}"
