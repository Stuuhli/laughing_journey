import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from typing import List

from utils import CHUNKS_DIR, CHROMA_DIR, get_embedding_object, get_chunk_dir_for_model

# Sliding window vs small to big.
# Teste Chunkgrößen 256, 512, 800
# HyDE w/ 1 pseudo-doc + query


def get_vector_store(embedding_model_name: str, persist: bool = True):
    """Create or load a Chroma vector store for an embedding model.

    Args:
        embedding_model_name (str): Name of the embedding model.
        persist (bool, optional): Persist data on disk if True. Defaults to True.

    Returns:
        Chroma: Vector store instance.
    """
    embeddings = get_embedding_object(embedding_model_name)
    model_dir = f"{CHROMA_DIR}/chroma__{embedding_model_name.replace(':', '-').replace('/', '-')}"
    persist_dir = model_dir if persist else None

    return Chroma(
        collection_name="docx_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )


def add_documents(documents, ids, vector_store):
    """Add new documents to the vector store.

    Args:
        documents (List[Document]): Documents to add.
        ids (List[str]): Corresponding document IDs.
        vector_store (Chroma): Target vector store.

    Returns:
        None
    """
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"[SUCCESS] Added {len(documents)} new documents")


def update_chroma_db(embedding_model_name: str, update: bool = False, filename: str = None, doc_path: Path = None):
    """Update the Chroma database with chunks from JSON files.

    Args:
        embedding_model_name (str): Embedding model name.
        update (bool, optional): Delete existing entries before adding new ones. Defaults to False.
        filename (str, optional): Specific chunk file to process.
        doc_path (str, optional): Path to a document whose chunks should be updated.

    Returns:
        None
    """
    vector_store = get_vector_store(embedding_model_name, persist=True)
    chunk_dir = get_chunk_dir_for_model(embedding_model_name)
    if doc_path:
        filenames = [Path(doc_path).name]
    elif filename:
        filenames = [filename]
    else:
        # Use .json files in CHUNKS_DIR instead
        # Fallback for the case in which no model-specific chunks have been created yet
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
        json_path = CHUNKS_DIR / fname.replace('.txt', '.json') if fname.endswith('.txt') else chunk_dir / fname
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


def delete_chroma_entry(chroma_id: str, embedding_model_name: str):
    """Delete a single entry from the Chroma database.

    Args:
        chroma_id (str): ID of the entry to remove.
        embedding_model_name (str): Embedding model name.
    """
    vector_store = get_vector_store(embedding_model_name, persist=True)
    print(f"[INFO] Delete chroma entry with ID: {chroma_id}")
    try:
        vector_store.delete(ids=[chroma_id])
        print(f"[SUCCESS] Entry {chroma_id} deleted.")
    except Exception as e:
        print(f"[ERROR] Error occured during deletion: {e}")


def create_chroma_db(embedding_model_name: str, doc_path=None):
    """Recreate the Chroma database for a specific model."""
    print("[INFO] Recreate Chroma-DB...")
    update_chroma_db(embedding_model_name=embedding_model_name, update=True, doc_path=doc_path)


def do_a_sim_search(query: str, k: int, vector_store: Chroma):
    """Perform a similarity search using a query string."""
    return vector_store.similarity_search(query, k=k)


def do_a_sim_search_with_embedding(embedding: List[float], k: int, vector_store: Chroma):
    """Perform a similarity search using a precomputed embedding."""
    return vector_store.similarity_search_by_vector(embedding=embedding, k=k)
