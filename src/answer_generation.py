from langchain_ollama import OllamaLLM
from vector_db import do_a_sim_search
from utils import (
    load_chunks,
    get_chunk_dir_for_model,
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_COMPARE,
    build_compare_context,
)
from rich.console import Console
import time
from langchain_chroma import Chroma

console = Console()


def _key(doc):
    """Create a stable identifier for a document based on metadata.

    Args:
        doc: Document object returned from the vector store.

    Returns:
        tuple: Source file and chapter path.
    """
    meta = doc.metadata
    return (
        meta.get("source_file", "Unknown file"),
        meta.get("chapter_path", "Unknown chapter"),
    )


def _prepare_prompt(
    query: str,
    k_values: int,
    vector_store: Chroma,
    embedding_model_name: str,
    use_full_chapters: bool,
):
    """Retrieve context and construct prompt and source appendix."""
    results = do_a_sim_search(query, k=k_values, vector_store=vector_store)

    unique_keys = []
    for doc in results:
        k = _key(doc)
        if k not in unique_keys:
            unique_keys.append(k)
    source_id_map = {k: i + 1 for i, k in enumerate(unique_keys)}

    print("--- Retrieved context ---")
    for idx, res in enumerate(results):
        snippet = res.page_content[:50].replace("\n", "")
        print(
            f"[{idx + 1}/{len(results)}] -> Chunk from '{res.metadata.get('source_file')}': {snippet}...\n"
            f"          -> Kapitelstruktur: {res.metadata.get('chapter_path')}"
        )
    print("-" * 25)

    if use_full_chapters:
        found_chapters = {doc.metadata.get("chapter_path") for doc in results if doc.metadata.get("chapter_path")}
        all_chunks = load_chunks(chunk_dir=get_chunk_dir_for_model(embedding_model_name))

        grouped = {}
        for c in all_chunks:
            k = (
                c["metadata"].get("source_file", "Unknown file"),
                c["metadata"].get("chapter_path", "Unknown chapter"),
            )
            if c["metadata"].get("chapter_path") in found_chapters and k in source_id_map:
                grouped.setdefault(k, []).append(c["page_content"])

        context_parts = []
        for k in unique_keys:
            if k in grouped:
                sid = source_id_map[k]
                combined = "\n".join(grouped[k])
                context_parts.append(f"[{sid}] {combined}")

        context_string = "\n\n--- DOC SEP ---\n\n".join(context_parts)

    else:
        grouped = {}
        for doc in results:
            k = _key(doc)
            grouped.setdefault(k, []).append(doc.page_content)

        context_parts = []
        for k in unique_keys:
            sid = source_id_map[k]
            combined = "\n".join(grouped[k])
            context_parts.append(f"[{sid}] {combined}")

        context_string = "\n\n--- DOC SEP ---\n\n".join(context_parts)

    prompt = PROMPT_TEMPLATE.format(context=context_string, query=query)

    sources_text = "\n\n---\nQuellen:\n"
    for k in unique_keys[:8]:
        sid = source_id_map[k]
        source_file, chapter_path = k
        sources_text += f"[{sid}] {source_file} – Kapitel: {chapter_path}\n"
    return prompt, sources_text


def generate_answer(
    model: str,
    query: str,
    k_values: int,
    vector_store: Chroma,
    embedding_model_name: str,
    use_full_chapters: bool = True,
):
    """Generate an answer using similarity search over the vector store."""

    start = time.time()
    prompt, sources_text = _prepare_prompt(
        query,
        k_values,
        vector_store,
        embedding_model_name,
        use_full_chapters,
    )

    llm = OllamaLLM(model=model, num_ctx=16384)
    with console.status(
        "[INFO] Request is being processed by the LLM ...",
        spinner="dots12",
        spinner_style="white",
    ):
        raw_response = llm.invoke(prompt)

    final_response_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

    final_response_text += sources_text

    print("Runtime: ", time.time() - start, " Sekunden")
    return final_response_text


def generate_answer_stream(
    model: str,
    query: str,
    k_values: int,
    vector_store: Chroma,
    embedding_model_name: str,
    use_full_chapters: bool = True,
):
    """Stream an answer token by token."""

    start = time.time()
    prompt, sources_text = _prepare_prompt(
        query,
        k_values,
        vector_store,
        embedding_model_name,
        use_full_chapters,
    )

    llm = OllamaLLM(model=model, num_ctx=16384)
    with console.status(
        "[INFO] Request is being processed by the LLM ...",
        spinner="dots12",
        spinner_style="white",
    ):
        for chunk in llm.stream(prompt):
            text = getattr(chunk, "content", None)
            if text is None:
                text = getattr(chunk, "text", str(chunk))
            yield text

    yield sources_text
    print("Runtime: ", time.time() - start, " Sekunden")


def generate_answer_compare_docs(
    model: str,
    query: str,
    selected_docx_paths,
    embedding_model_name: str,
):
    """Generate an answer by comparing multiple complete documents.

    Args:
        model (str): Generation model to use.
        query (str): User question.
        selected_docx_paths: Paths to the documents that should be compared.
        embedding_model_name (str): Embedding model used for chunk retrieval.

    Returns:
        str: LLM response including source list.
    """
    start = time.time()

    # 1) Nummerierten Vollkontext bauen
    context_string, sources = build_compare_context(selected_docx_paths, embedding_model_name)
    if not context_string:
        return "[ERROR] No context could be created. Please check DOCX_DIR/CONVERTED_DIR/Chunks."

    # 2) Antwort generieren (Vergleichsprompt)
    llm = OllamaLLM(model=model, num_ctx=16384)
    prompt = PROMPT_TEMPLATE_COMPARE.format(context=context_string, query=query)
    with console.status("[INFO] Comparing documents via LLM ...", spinner="dots12", spinner_style="white"):
        raw_response = llm.invoke(prompt)

    final_response_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

    # 3) Quellenverzeichnis (IDs entsprechen Reihenfolge der ausgewählten docx)
    sources_text = "\n\n---\nQuellen:\n"
    for sid, label in sources:
        sources_text += f"[{sid}] {label}\n"

    final_response_text += sources_text

    print("Runtime: ", time.time() - start, " seconds")
    return final_response_text
