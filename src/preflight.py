import os
from docling_converter import convert_all_docx
from chunking import create_chunks_for_all
from vector_db import update_chroma_db, delete_chroma_entry
from utils import CONVERTED_DIR, CHUNKS_DIR, CHROMA_DIR


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
    if os.path.exists(CHROMA_DIR):
        print("Chroma directory exists.")
        action = input("Update chroma (a), delete by ID (l), create new (n), skip (s)? ").strip().lower()
        if action == 'a':
            update_chroma_db(update=True)
        elif action == 'l':
            chroma_id = input("ID to delete: ").strip()
            delete_chroma_entry(chroma_id)
            print(f"Entry {chroma_id} deleted.")
        elif action == 'n':
            update_chroma_db(update=False)
        else:
            print("Skipping chroma.")
    else:
        print("Chroma directory does not exist. Starting creation.")
        os.makedirs(CHROMA_DIR, exist_ok=True)
        update_chroma_db(update=False)


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
    # The following imports are scoped locally in the original, so we keep them here.
    from utils import CONVERTED_DIR
    import os
    txt_filename = os.path.splitext(os.path.basename(clean_path))[0] + ".txt"
    txt_path = str(CONVERTED_DIR / txt_filename)
    create_chunks_for_all(doc_path=txt_path)
    update_chroma_db(doc_path=txt_path)
    print("--- [PIPELINE] Pipeline finished. ---")
