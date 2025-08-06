from answer_generation import generate_answer
from utils import GENERATION_MODELS, EMBEDDING_MODELS, get_models_from_user, get_k_values_from_user, generic_let_user_choose
from preflight import preflight, run_full_pipeline_for_new_doc
from vector_db import get_vector_store


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
            vector_store = get_vector_store(embedding_model_name=embedding_model_name, persist=True)
            while True:
                try:
                    query = input("🔍 Enter your query ('q' to quit): ").strip()
                    if not query or query.lower() == 'q':
                        print("[INFO] Program aborted")
                        break

                    model_to_run = get_models_from_user(available_models=GENERATION_MODELS)[0]
                    k_values = get_k_values_from_user()[0]
                    response = generate_answer(
                        model=model_to_run,
                        query=query,
                        k_values=k_values,
                        vector_store=vector_store,
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
