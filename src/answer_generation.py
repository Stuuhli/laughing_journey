from langchain_ollama import OllamaLLM
from vector_db import do_a_sim_search
from utils import load_chunks, get_chunk_dir_for_model
from rich.console import Console
import time
from langchain_chroma import Chroma
from utils import PROMPT_TEMPLATE

console = Console()


def generate_answer(
    model: str,
    query: str,
    k_values: int,
    vector_store: Chroma,
    embedding_model_name: str,
    use_full_chapters: bool = True   # Toggle für Kontextmodus
):
    start = time.time()

    # 1) Retrieval
    results = do_a_sim_search(query, k=k_values, vector_store=vector_store)

    print("--- Retrieved context ---")
    for idx, res in enumerate(results):
        snippet = res.page_content[:50].replace("\n", "")
        print(
            f"[{idx + 1}/{len(results)}] -> Chunk from '{res.metadata.get('source_file')}': {snippet}...\n"
            f"          -> Kapitelstruktur: {res.metadata.get('chapter_path')}"
        )
    print("-" * 25)

    # 2) Kontext vorbereiten
    if use_full_chapters:
        # Small-to-Big: alle Chunks der Kapitel, in denen Treffer liegen
        found_chapters = set()
        for doc in results:
            chapter_tag = doc.metadata.get("chapter_path")
            if chapter_tag:
                found_chapters.add(chapter_tag)

        all_chunks = load_chunks(chunk_dir=get_chunk_dir_for_model(embedding_model_name))
        chapter_chunks = [
            c for c in all_chunks if c["metadata"].get("chapter_path") in found_chapters
        ]
        chapter_chunks = sorted(
            chapter_chunks,
            key=lambda c: (
                c["metadata"].get("source_file", ""),
                c["metadata"].get("Chapter_number", "0")
            )
        )
        context_string = "\n\n--- DOC SEP ---\n\n".join([c["page_content"] for c in chapter_chunks])

    else:
        # Nur Top-K direkte Treffer
        context_string = "\n\n--- DOC SEP ---\n\n".join([doc.page_content for doc in results])

    # 3) Antwort generieren (ein Pass)
    llm = OllamaLLM(model=model, num_ctx=16384)
    prompt = PROMPT_TEMPLATE.format(context=context_string, query=query)
    with console.status("[INFO] Request is being processed by the LLM ...",
                        spinner='dots12', spinner_style='white'):
        raw_response = llm.invoke(prompt)

    final_response_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

    # 4) Quellenliste aus Retrieval-Ergebnis (immer Top-K Treffer)
    sources_text = "\n\n---\nQuellen:\n"
    for idx, doc in enumerate(results[:8]):
        meta = doc.metadata
        source_file = meta.get("source_file", "Unbekannte Datei")
        chapter_path = meta.get("chapter_path", "Unbekanntes Kapitel")
        sources_text += f"[{idx + 1}] {source_file} – Kapitel: {chapter_path}\n"

    final_response_text += sources_text

    print("Runtime: ", time.time() - start, " Sekunden")
    return final_response_text
