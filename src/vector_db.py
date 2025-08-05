import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from utils import CHUNKS_DIR, CHROMA_DIR

embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2:568m")

vector_store = Chroma(
    collection_name="docx_collection",
    embedding_function=embeddings,
    # if commented out, vector db is stored in RAM -> good for quick tests, but db has to be rebuilt for every iteration
    # which is only good when testing different chunk_sizes, which is not yet implemented
    # (currently only fixed/predefined chunks)
    persist_directory=str(CHROMA_DIR)
)


def add_documents(documents, ids):
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"[SUCCESS] Added {len(documents)} new documents")


def update_chroma_db(update=False, filename=None, doc_path=None):
    """
    Updates all chunks from a specific source file in the vector store.
    It deletes the old chunks and adds the new ones from the corresponding .json file.

    Args:
        filename (str): The name of the source file (e.g., "KISTERS_IS_PasswortRichtlinie.txt").
        vector_store (Chroma): The Chroma vector store instance.
    """
    """
    Updates Chroma-DB for a specific document or all documents. Optional: Update mode.
    """
    if doc_path:
        filenames = [Path(doc_path).name]
    elif filename:
        filenames = [filename]
    else:
        # Alle .json Dateien im CHUNKS_DIR
        filenames = [f.name for f in CHUNKS_DIR.glob("*.json")]

    for fname in filenames:
        print(f"[INFO] Starting update for: {fname}")
        # 1. Identify and delete old chunks based on filename 🗑️
        source_txt = fname.replace('.json', '.txt')
        print(f"[INFO] [1/3] Searching for old chunks with source_file='{source_txt}'...")
        existing_docs = vector_store.get(where={"source_file": source_txt})
        ids_to_delete = existing_docs['ids']
        if update and ids_to_delete:
            print(f"[INFO] -> Found {len(ids_to_delete)} old chunks. Deleting...")
            vector_store.delete(ids=ids_to_delete)
            print("[SUCCESS] -> Deletion successful.")
        else:
            print("[INFO] -> No old chunks found to delete or update not enabled.")
        # 2. Load new chunks from the corresponding JSON file ⚙️
        print("[INFO] [2/3] Loading new chunks from generated JSON file...")
        json_path = CHUNKS_DIR / fname.replace('.txt', '.json') if fname.endswith('.txt') else CHUNKS_DIR / fname
        if not json_path.exists():
            print(f"[ERROR] -> New chunk file not found at: {json_path}")
            print("[ERROR] -> Please run the chunking script for the updated file first.")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            new_chunks_data = json.load(f)
        if not new_chunks_data:
            print("[INFO] -> JSON file is empty. No new chunks to add.")
            print(f"[SUCCESS] Update process for {fname} finished.")
            continue
        print(f"[SUCCESS] -> Found and loaded {len(new_chunks_data)} new chunks.")
        # 3. Add new chunks to the database ✅
        print("[INFO] [3/3] Adding new chunks to the database...")
        new_docs = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in new_chunks_data]
        new_ids = [c["metadata"]["chunk_id"] for c in new_chunks_data]
        vector_store.add_documents(documents=new_docs, ids=new_ids)
        print(f"[SUCCESS] -> Added {len(new_docs)} new chunks to the database.")
        print(f"[SUCCESS] Update for {fname} completed.")


def delete_chroma_entry(chroma_id: str):
    """
    Deletes an entry from the Chroma DB based on the chunk_id.
    """
    print(f"[INFO] Delete chroma entry with ID: {chroma_id}")
    try:
        vector_store.delete(ids=[chroma_id])
        print(f"[SUCCESS] Entry {chroma_id} deleted.")
    except Exception as e:
        print(f"[ERROR] Error occured during deletion: {e}")


def create_chroma_db(doc_path=None):
    """
    Recreates the Chroma DB from all or a specific document.
    """
    print("[INFO] Recreate Chroma-DB...")
    update_chroma_db(update=True, doc_path=doc_path)


def do_a_sim_search(query: str, k: int):
    return vector_store.similarity_search(query, k=k)
