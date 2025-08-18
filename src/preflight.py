import os
from docling_converter import convert_all_docx
from chunking import create_chunks_for_all
from vector_db import update_chroma_db
from utils import CONVERTED_DIR, CHUNKS_DIR, CHROMA_DIR, get_models_from_user, EMBEDDING_MODELS, EMBEDDING_MAX_LENGTHS, get_chunk_dir_for_model
import shutil
import stat

# TODO: Make it possible to generate the base-chunks with only chapter-wise splits
# TODO: Change Chunk-structure from model-based to context-size-based => more rigid and dynamic, avoids redundancy


def _remove_readonly(func, path, _):
    """Helper funciton: remove read-only-flag and try again"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def check_and_manage_converted():
    """Handle converted text files (update, delete or create)."""
    print("\n--- [PRE] Check: Converted documents ---")

    if os.path.exists(CONVERTED_DIR):
        existing_files = os.listdir(CONVERTED_DIR)
        print("Converted-Verzeichnis:")
        for f in existing_files:
            print(f"  - {f}")
    else:
        print("Converted directory does not exist. Creating it now...")
        os.makedirs(CONVERTED_DIR, exist_ok=True)

    print("\nOptionen:")
    print("  [a] Alle Dokumente aktualisieren/ergänzen")
    print("  [d] Alle konvertierten Dokumente löschen")
    print("  [n] Neues Dokument konvertieren")
    print("  [s] Überspringen")

    action = input("Aktion wählen: ").strip().lower()

    if action == 'a':
        convert_all_docx(update=True)
    elif action == 'd':
        for f in os.listdir(CONVERTED_DIR):
            os.remove(os.path.join(CONVERTED_DIR, f))
        print("[SUCCESS] Alle konvertierten Dokumente gelöscht.")
    elif action == 'n':
        doc_path = input("Pfad zum neuen Dokument eingeben: ").strip().strip('"')
        convert_all_docx(doc_path=doc_path)
        print(f"[SUCCESS] Dokument konvertiert: {doc_path}")
    else:
        print("Skipping converted.")


def check_and_manage_chunks():
    """Manage chunk directories for different embedding models."""
    print("\n--- [PRE] Check: Chunks ---")
    chunk_base = CHUNKS_DIR
    existing_chunk_dirs = [d for d in os.listdir(chunk_base) if d.startswith("chunks__")]

    print("Chunks-Verzeichnisse:")
    for d in existing_chunk_dirs:
        print(f"  - {d}")

    print("\nOptionen:")
    print("  [a] Alle Chunks für alle Modelle neu erzeugen")
    print("  [d] Einzelnen Chunk-Ordner löschen")
    print("  [n] Neue Chunks für ein Modell erzeugen")
    print("  [s] Überspringen")

    action = input("Aktion wählen: ").strip().lower()
    if action == 'a':
        for emb in EMBEDDING_MODELS:
            max_length = EMBEDDING_MAX_LENGTHS[emb]
            chunk_dir = get_chunk_dir_for_model(emb)
            print(f"[INFO] Erzeuge Chunks für {emb} (max_length={max_length}) im Ordner {chunk_dir}")
            create_chunks_for_all(update=True, max_chunk_length=max_length, chunk_dir=chunk_dir)

    elif action == 'd':
        if not existing_chunk_dirs:
            print("[WARN] Keine Chunk-Ordner vorhanden.")
            return
        print("Wähle zu löschenden Chunk-Ordner:")
        for i, d in enumerate(existing_chunk_dirs):
            print(f"  [{i + 1}] {d}")
        idx = int(input("> ")) - 1
        if 0 <= idx < len(existing_chunk_dirs):
            dir_to_delete = os.path.join(chunk_base, existing_chunk_dirs[idx])
            shutil.rmtree(dir_to_delete, onerror=_remove_readonly)
            print(f"[SUCCESS] Chunk-Ordner gelöscht: {dir_to_delete}")
        else:
            print("[WARN] Ungültige Auswahl.")

    elif action == 'n':
        if not existing_chunk_dirs:
            print("[WARN] Keine Chunk-Ordner vorhanden. Bitte zuerst 'a' wählen, um welche zu erzeugen.")
            return
        print("Wähle Chunk-Ordner für neue Chunks:")
        for i, d in enumerate(existing_chunk_dirs):
            print(f"  [{i + 1}] {d}")
        idx = int(input("> ")) - 1
        if 0 <= idx < len(existing_chunk_dirs):
            # Embedding-Name aus Ordner ableiten
            emb = existing_chunk_dirs[idx].replace("chunks__", "").replace("-", "/")
            max_length = EMBEDDING_MAX_LENGTHS.get(emb, 512)
            chunk_dir = get_chunk_dir_for_model(emb)
            print(f"[INFO] Erzeuge Chunks für {emb} (max_length={max_length}) im Ordner {chunk_dir}")
            create_chunks_for_all(update=True, max_chunk_length=max_length, chunk_dir=chunk_dir)
        else:
            print("[WARN] Ungültige Auswahl.")

    else:
        print("Skipping chunks.")


def check_and_manage_chroma():
    """Manage local Chroma vector stores."""
    print("\n--- [PRE] Check: Chroma DB ---")
    chroma_base = CHROMA_DIR

    # nur tatsächlich vorhandene Chroma-DBs
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
        if not existing_dbs:
            print("[WARN] Keine Chroma-DBs vorhanden.")
            return
        print("Wähle zu löschende DB:")
        for i, db in enumerate(existing_dbs):
            print(f"  [{i + 1}] {db}")
        idx = int(input("> ")) - 1
        if 0 <= idx < len(existing_dbs):
            dir_to_delete = os.path.join(chroma_base, existing_dbs[idx])
            shutil.rmtree(dir_to_delete, onerror=_remove_readonly)
            print(f"[SUCCESS] Chroma-DB gelöscht: {dir_to_delete}")
        else:
            print("[WARN] Ungültige Auswahl.")

    elif action == 'n':
        # hier wie bei Chunks nur aus verfügbaren wählen
        print("Wähle Modell für neue/aktualisierte Chroma-DB:")
        for i, emb in enumerate(EMBEDDING_MODELS):
            print(f"  [{i + 1}] {emb}")
        idx = int(input("> ")) - 1
        if 0 <= idx < len(EMBEDDING_MODELS):
            embedding_model_name = EMBEDDING_MODELS[idx]
            update_chroma_db(embedding_model_name=embedding_model_name, update=True)
        else:
            print("[WARN] Ungültige Auswahl.")
    else:
        print("Skipping chroma.")


def preflight():
    """Run all preflight checks."""
    print("\n--- [PRE] Starting preflight... ---")
    check_and_manage_converted()
    check_and_manage_chunks()
    check_and_manage_chroma()
    print("--- [PRE] Preflight finished. ---")


def run_full_pipeline_for_new_doc(doc_path):
    """Convert, chunk and index a new document.

    Args:
        doc_path: Path to the document that should be processed.
    """
    clean_path = str(doc_path).strip().strip('"')
    print(f"\n--- [PIPELINE] Starting full pipeline for: {clean_path} ---")
    convert_all_docx(doc_path=clean_path)
    txt_filename = os.path.splitext(os.path.basename(clean_path))[0] + ".txt"
    txt_path = str(CONVERTED_DIR / txt_filename)
    embedding_model_name = get_models_from_user(available_models=EMBEDDING_MODELS, test_mode=False)[0]
    max_length = EMBEDDING_MAX_LENGTHS[embedding_model_name]
    chunk_dir = get_chunk_dir_for_model(embedding_model_name)
    create_chunks_for_all(doc_path=txt_path, update=True, max_chunk_length=max_length, chunk_dir=chunk_dir)
    update_chroma_db(embedding_model_name=embedding_model_name, doc_path=txt_path)
    print("--- [PIPELINE] Pipeline finished. ---")
