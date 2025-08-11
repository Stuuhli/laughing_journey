from answer_generation import generate_answer, generate_answer_compare_docs
from utils import (
    GENERATION_MODELS,
    EMBEDDING_MODELS,
    get_models_from_user,
    get_k_values_from_user,
    generic_let_user_choose,
    list_docx_files,
)
from preflight import preflight, run_full_pipeline_for_new_doc
from vector_db import get_vector_store


def _choose_chat_mode():
    opts = ["Similarity Search (RAG über Chroma)", "Vergleich mehrerer Dokumente (vollständiger Kontext)"]
    choice = generic_let_user_choose(
        prompt="Wähle den Chat-Modus:",
        options=opts,
        allow_multiple=False
    )[0]
    return "sim" if choice == opts[0] else "compare"


def _choose_docx_multi():
    files = list_docx_files()
    if not files:
        print("[ERROR] Keine .docx-Dateien in DOCX_DIR gefunden.")
        return []

    # Drucke nur Namen
    options = [f.name for f in files]
    selection = generic_let_user_choose(
        prompt="Wähle ein oder mehrere .docx-Dokumente (z. B. 1,3,5):",
        options=options,
        allow_multiple=True
    )

    # Map zurück auf Pfade
    idxs = [options.index(name) for name in selection]
    return [files[i] for i in idxs]


def main():
    print("Welcome to the Document and Chatbot Tool!")
    while True:
        print("\n[1] Run preflight check\n[2] Run full pipeline for new document\n[3] Start chatbot\n[q] Quit")
        choice = input("Selection: ").strip().lower()
        if choice == '1':
            print("Running preflight check...")
            preflight()
        elif choice == '2':
            doc_path = input("Path to new document: ").strip()
            print(f"Running full pipeline for: {doc_path}")
            run_full_pipeline_for_new_doc(doc_path)
        elif choice == '3':
            print("Chatbot started. Type your query below.")

            embedding_model_name = generic_let_user_choose(
                prompt="Which embedding model should be used for the search?",
                options=EMBEDDING_MODELS,
                allow_multiple=False
            )[0]

            chat_mode = _choose_chat_mode()

            # Für Similarity Search benötigen wir die Vector-DB
            vector_store = None
            if chat_mode == "sim":
                vector_store = get_vector_store(embedding_model_name=embedding_model_name, persist=True)

            while True:
                try:
                    query = input("🔍 Enter your query ('q' to quit): ").strip()
                    if not query or query.lower() == 'q':
                        print("[INFO] Program aborted")
                        break

                    model_to_run = get_models_from_user(available_models=GENERATION_MODELS)[0]

                    if chat_mode == "sim":
                        k_values = get_k_values_from_user()[0]
                        response = generate_answer(
                            model=model_to_run,
                            query=query,
                            k_values=k_values,
                            vector_store=vector_store,
                            embedding_model_name=embedding_model_name
                        )
                    else:
                        # Vergleichsmodus: mehrere Dokx wählen, Kontext bauen, vergleichen
                        selected_docs = _choose_docx_multi()
                        if not selected_docs:
                            print("[WARN] Keine Dokumente ausgewählt.")
                            continue
                        response = generate_answer_compare_docs(
                            model=model_to_run,
                            query=query,
                            selected_docx_paths=selected_docs,
                            embedding_model_name=embedding_model_name
                        )

                    print("LLM Response:")
                    print(response)

                except KeyboardInterrupt:
                    print("\n[INFO] Program aborted")
                    break
        elif choice == 'q':
            print("[INFO] Program aborted")
            break
        else:
            print("Invalid selection. Please try again.")


if __name__ == "__main__":
    main()
