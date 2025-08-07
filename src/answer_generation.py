from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from vector_db import do_a_sim_search, do_a_sim_search_with_embedding
from utils import PROMPT_TEMPLATE, get_embedding_object, get_search_method_from_user, load_chunks, get_chunk_dir_for_model
from rich.console import Console
import time
from langchain_chroma import Chroma


console = Console()


def generate_answer(model: str, query: str, k_values: int, vector_store: Chroma, embedding_model_name: str):
    start = time.time()

    search_method = get_search_method_from_user()

    prompt = ChatPromptTemplate.from_template(template=PROMPT_TEMPLATE)
    llm = OllamaLLM(model=model, num_ctx=16384)
    chain = prompt | llm

    results = []

    if search_method == "hyde":
        embedding_model = get_embedding_object(embedding_model_name)
        hyde_chat_model = ChatOllama(model="phi3:3.8b")
        hyde_embedding = get_hypo_embedding(query=query, embedding_model=embedding_model, chat_model=hyde_chat_model)
        results = do_a_sim_search_with_embedding(embedding=hyde_embedding, k=k_values, vector_store=vector_store)

        print(f"\n[INFO] Search with HyDe method (model: {embedding_model_name})")

    else:
        results = do_a_sim_search(query, k=k_values, vector_store=vector_store)
        print("\n[INFO] Search with normal method")

    # Debug-Ausgabe: Zeigt an, was gefunden wurde
    print("--- Retrieved context ---")
    for idx, res in enumerate(results):
        # Kurze Ausgabe des Inhalts und der Quelle
        print(f"[{idx + 1}/{len(results)}] -> Chunk from '{res.metadata.get('source_file')}': {res.page_content[:80]}...")
    print("-" * 25)

    # context_string = "\n\n---\n\n".join([doc.page_content for doc in results])
    found_chapters = set()
    for doc in results:
        chapter_tag = doc.metadata.get("chapter_path")
        if chapter_tag:
            found_chapters.add(chapter_tag)
    all_chunks = load_chunks(chunk_dir=get_chunk_dir_for_model(embedding_model_name))
    chapter_chunks = []
    for chapter_tag in found_chapters:
        chapter_chunks.extend([c for c in all_chunks if c["metadata"].get("chapter_path") == chapter_tag])

    chapter_chunks = sorted(
        chapter_chunks,
        key=lambda c: (
            c["metadata"].get("source_file", ""),
            c["metadata"].get("Chapter_number", "0")
        )
    )
    context_string = "\n\n---\n\n".join([c["page_content"] for c in chapter_chunks])

    with console.status("[INFO] Request is being processed by the LLM ...", spinner='dots12', spinner_style='white'):
        response = chain.invoke({
            "context": context_string,
            "query": query
        })

    print("Runtime: ", time.time() - start, " Sekunden")
    return response


def get_hypo_embedding(query: str, embedding_model, chat_model):
    hypo_doc = get_hypo_doc(query, chat_model)
    hypo_embedding = embedding_model.embed_documents([hypo_doc])
    return hypo_embedding[0]


def get_hypo_doc(query, chat_model):
    template = """Stell dir vor, du bist ein Experte, der eine ausführliche Erklärung zum Thema '{query}' verfasst.
    Deine Antwort sollte umfasd sein und alle wichtigen Punkte enthalten, die in den obersten Suchergebnissen zu finden sind."""

    system_message_prompt = SystemMessagePromptTemplate.from_template(template=template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
    messages = chat_prompt.format_prompt(query=query).to_messages()

    response = chat_model.invoke(messages)
    hypo_doc = response.content

    print("\n--- Generierte hypothetische Antwort (HyDE) ---")
    print(hypo_doc)
    print("-" * 40)

    return hypo_doc
