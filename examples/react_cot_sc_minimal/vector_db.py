from langchain_core.documents import Document
from typing import List

class InMemoryVectorStore:
    """Very small in-memory store used for the minimal ReAct example."""
    def __init__(self, docs: List[Document]):
        self.docs = docs

    def similarity_search(self, query: str, k: int = 5):
        return self.docs[:k]

def get_vector_store(embedding_model_name: str, persist: bool = True):
    """Return an in-memory vector store populated with demo documents."""
    docs = [
        Document(page_content="Das ist ein Beispieldokument über Informationssicherheit."),
        Document(page_content="Ein weiteres Dokument erklärt Business Continuity."),
    ]
    return InMemoryVectorStore(docs)

def do_a_sim_search(query: str, k: int, vector_store: InMemoryVectorStore):
    """Run a naive similarity search over the in-memory documents."""
    return vector_store.similarity_search(query, k=k)
