from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from vector_db import do_a_sim_search
from utils import PROMPT_TEMPLATE
from rich.console import Console
import time

console = Console()


def generate_answer(model: str, query: str, k_values: int):
    start = time.time()
    prompt = ChatPromptTemplate.from_template(template=PROMPT_TEMPLATE)
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
    gemma:2b:           8192
    """
    llm = OllamaLLM(model=model, num_ctx=4096)

    # langchain expression language LCEL
    chain = prompt | llm

    results = do_a_sim_search(query, k=k_values)

    # Debug-Ausgabe: Zeigt an, was gefunden wurde
    print("--- Gefundener Kontext ---")
    for idx, res in enumerate(results):
        # Kurze Ausgabe des Inhalts und der Quelle
        print(f"[{idx + 1}/{len(results)}] -> Chunk aus '{res.metadata.get('source_file')}': {res.page_content[:80]}...")
    print("-" * 25)

    context_string = "\n\n---\n\n".join([doc.page_content for doc in results])

    with console.status("[INFO] Anfrage wird von LLM bearbeitet ...", spinner='dots12', spinner_style='white'):
        response = chain.invoke({
            "context": context_string,
            "query": query
        })

    print("Programmdauer: ", time.time() - start, " Sekunden")
    return response
