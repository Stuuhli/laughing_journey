import os
from docling_converter import convert_all_docx
from chunking import create_chunks_for_all
from vector_db import update_chroma_db
from utils import CONVERTED_DIR, CHUNKS_DIR, CHROMA_DIR, get_models_from_user, EMBEDDING_MODELS
import shutil


def check_and_manage_converted():
    print("\n--- [PRE] Check: Converted documents ---")
    if os.path.exists(CONVERTED_DIR):
        print("Converted directory exists.")
        action = input("Update documents (a), delete (l), convert new (n), skip (s)? ").strip().lower()
        if action == 'a':
            convert_all_docx(update=True)
        elif action == 'l':
            for f in os.listdir(CONVERTED_DIR):
                os.remove(os.path.join(CONVERTED_DIR, f))
            print("All converted documents deleted.")
        elif action == 'n':
            convert_all_docx(update=False)
        else:
            print("Skipping converted.")
    else:
        print("Converted directory does not exist. Starting conversion.")
        os.makedirs(CONVERTED_DIR, exist_ok=True)
        convert_all_docx(update=False)


def check_and_manage_chunks():
    print("\n--- [PRE] Check: Chunks ---")
    if os.path.exists(CHUNKS_DIR):
        print("Chunks directory exists.")
        action = input("Update chunks (a), delete (l), create new (n), skip (s)? ").strip().lower()
        if action == 'a':
            create_chunks_for_all(update=True)
        elif action == 'l':
            for f in os.listdir(CHUNKS_DIR):
                os.remove(os.path.join(CHUNKS_DIR, f))
            print("All chunks deleted.")
        elif action == 'n':
            create_chunks_for_all(update=False)
        else:
            print("Skipping chunks.")
    else:
        print("Chunks directory does not exist. Starting creation.")
        os.makedirs(CHUNKS_DIR, exist_ok=True)
        create_chunks_for_all(update=False)


def check_and_manage_chroma():
    print("\n--- [PRE] Check: Chroma DB ---")
    chroma_base = CHROMA_DIR

    # Liste aller existierenden Modellordner finden:
    existing_dbs = [d for d in os.listdir(chroma_base) if d.startswith("chroma__")]

    print("Chroma-Verzeichnisse:")
    for db in existing_dbs:
        print(f"  - {db}")

    print("\nOptionen:")
    print("  [a] Alle Chroma-DBs aktualisieren")
    print("  [d] Einzelne Chroma-DB löschen")
    print("  [n] Neue Chroma-DB anlegen/aktualisieren (für ein Modell)")
    print("  [s] Überspringen")

    action = input("Aktion wählen: ").strip().lower()
    if action == 'a':
        for emb in EMBEDDING_MODELS:
            print(f"[INFO] Update für {emb} ...")
            update_chroma_db(embedding_model_name=emb, update=True)
    elif action == 'd':
        print("Wähle zu löschende DB (Embedding-Modell):")
        for i, emb in enumerate(EMBEDDING_MODELS):
            print(f"  [{i + 1}] {emb}")
        idx = int(input("> ")) - 1
        if 0 <= idx < len(EMBEDDING_MODELS):
            emb = EMBEDDING_MODELS[idx]
            db_dir = f"{chroma_base}/chroma__{emb.replace(':', '-').replace('/', '-')}"
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)
                print(f"[SUCCESS] Chroma-DB für {emb} gelöscht: {db_dir}")
            else:
                print(f"[WARN] Ordner nicht gefunden: {db_dir}")
    elif action == 'n':
        embedding_model_name = get_models_from_user(available_models=EMBEDDING_MODELS, test_mode=False)[0]
        update_chroma_db(embedding_model_name=embedding_model_name, update=True)
    else:
        print("Skipping chroma.")


def preflight():
    print("\n--- [PRE] Starting preflight... ---")
    check_and_manage_converted()
    check_and_manage_chunks()
    check_and_manage_chroma()
    print("--- [PRE] Preflight finished. ---")


def run_full_pipeline_for_new_doc(doc_path):
    clean_path = str(doc_path).strip().strip('"')
    print(f"\n--- [PIPELINE] Starting full pipeline for: {clean_path} ---")
    convert_all_docx(doc_path=clean_path)
    txt_filename = os.path.splitext(os.path.basename(clean_path))[0] + ".txt"
    txt_path = str(CONVERTED_DIR / txt_filename)
    create_chunks_for_all(doc_path=txt_path)
    embedding_model_name = get_models_from_user(available_models=EMBEDDING_MODELS, test_mode=False)[0]
    update_chroma_db(embedding_model_name=embedding_model_name, doc_path=txt_path)
    print("--- [PIPELINE] Pipeline finished. ---")
