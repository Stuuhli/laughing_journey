"""Automatic routing between semantic search and document analysis using LangChain tools."""
from typing import List

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from answer_generation import generate_answer, generate_answer_compare_docs
from vector_db import get_vector_store
from utils import list_docx_files


@tool(
    "semantic_search_tool",
    description=(
        "Perform semantic search over indexed document chunks. "
        "Use this when the question can be answered by retrieving "
        "specific passages from the corpus."
    ),
)
def semantic_search_tool(query: str, model: str, k: int, embedding_model_name: str) -> str:
    """Answer a question by performing semantic search over indexed documents."""
    vector_store = get_vector_store(embedding_model_name=embedding_model_name, persist=True)
    return generate_answer(
        model=model,
        query=query,
        k_values=k,
        vector_store=vector_store,
        embedding_model_name=embedding_model_name,
    )


@tool(
    "document_context_tool",
    description=(
        "Load whole documents and supply them as context for the answer. "
        "Use this when the query requires analysing an entire document or "
        "comparing multiple documents, for example compliance checks. "
        "Optional parameter 'documents' accepts a list of document names; "
        "leave empty to use all available documents."
    ),
)
def document_context_tool(
    query: str, model: str, documents: List[str], embedding_model_name: str
) -> str:
    """Use complete documents as context to answer the question.

    Parameters
    ----------
    query : str
        The user question.
    model : str
        Generation model to use.
    documents : List[str]
        List of document names to load. If empty, all available documents are used.
    embedding_model_name : str
        Name of the embedding model for chunk retrieval.
    """
    available_docs = list_docx_files()
    selected = []
    lower_map = {p.name.lower(): p for p in available_docs}
    for name in documents or []:
        key = name.lower()
        if key in lower_map:
            selected.append(lower_map[key])
    if not selected:
        selected = available_docs
    return generate_answer_compare_docs(
        model=model,
        query=query,
        selected_docx_paths=selected,
        embedding_model_name=embedding_model_name,
    )


def answer_query_with_tools(query: str, model: str, embedding_model_name: str, k: int = 5) -> str:
    """Route the query through an LLM with function calling support."""
    tools = [semantic_search_tool, document_context_tool]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    ai_message = llm_with_tools.invoke([HumanMessage(query)])
    if ai_message.tool_calls:
        call = ai_message.tool_calls[0]
        if call["name"] == "semantic_search_tool":
            return semantic_search_tool.invoke(
                {
                    "query": query,
                    "model": model,
                    "k": k,
                    "embedding_model_name": embedding_model_name,
                }
            )
        if call["name"] == "document_context_tool":
            documents = call["args"].get("documents", [])
            return document_context_tool.invoke(
                {
                    "query": query,
                    "model": model,
                    "documents": documents,
                    "embedding_model_name": embedding_model_name,
                }
            )
    return ai_message.content
