"""ReAct-CoT-SC agent implementation.

This module provides an example implementation of the
Reason + Act (ReAct) paradigm enhanced with Chain of Thought (CoT)
reasoning and a simple self-consistency check (SC).

The agent accepts a user query and lets a reasoning model decide
which action to take:

* ``similarity_search`` – run a semantic search over indexed chunks
  and use the retrieved passages as context.
* ``retrieve_file`` – load whole documents and use them as context.

After executing the chosen action, the agent verifies via another
reasoning step whether the gathered context is sufficient to answer
the question.  If it is, a final answer is generated.  If not, the
agent performs a second iteration with a newly reasoned action.

The code is intentionally lightweight and serves as a blueprint for
integrating ReAct with CoT and a basic self‑consistency loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.messages import HumanMessage

from vector_db import get_vector_store, do_a_sim_search
from utils import PROMPT_TEMPLATE, list_docx_files, build_compare_context


@dataclass
class ReActDecision:
    """Represents a decision made by the reasoning model."""

    action: str
    query: str


class ReActCoTSCAgent:
    """Minimal ReAct agent with Chain of Thought and self‑consistency."""

    def __init__(self, model: str, embedding_model_name: str, k: int = 5) -> None:
        self.model = model
        self.embedding_model_name = embedding_model_name
        self.k = k
        # Reasoning model is also used for final answer generation
        self.reasoner = ChatOllama(model=model, temperature=0, num_ctx=131072)
        self.vector_store = get_vector_store(
            embedding_model_name=embedding_model_name, persist=True
        )

    # ------------------------------------------------------------------
    # Reasoning helpers
    # ------------------------------------------------------------------
    def _think(self, prompt: str) -> str:
        """Run the reasoning model with a CoT style instruction."""
        cot_prompt = (
            "Denke Schritt für Schritt über das Problem nach und begründe deine Antwort.\n"
            + prompt
        )
        print("\n[DEBUG] Prompt an das Reasoning-Modell:\n", cot_prompt)
        msg = self.reasoner.invoke([HumanMessage(content=cot_prompt)])
        response = msg.content if hasattr(msg, "content") else str(msg)
        print("[DEBUG] Ausgabe des Reasoning-Modells:\n", response)
        return response

    def _decide_action(self, query: str) -> ReActDecision:
        """Let the model choose an action and optionally reformulate the query."""
        prompt = (
            "Du kannst zwei Aktionen ausführen: "
            "'similarity_search' für semantische Suche oder 'retrieve_file' zum Laden ganzer Dokumente.\n"
            "Gib deine Entscheidung im Format:\n"
            "ACTION: <similarity_search|retrieve_file>\n"
            "QUERY: <formulierte Suchanfrage oder leer>"
        )
        response = self._think(f"{prompt}\nBenutzeranfrage: {query}")
        action_line = next(
            (l for l in response.splitlines() if l.lower().startswith("action:")),
            "ACTION: similarity_search",
        )
        query_line = next(
            (l for l in response.splitlines() if l.lower().startswith("query:")),
            f"QUERY: {query}",
        )
        action = "retrieve_file" if "retrieve_file" in action_line.lower() else "similarity_search"
        reformulated = query_line.split(":", 1)[1].strip()
        return ReActDecision(action=action, query=reformulated or query)

    def _is_context_sufficient(self, query: str, context: str) -> bool:
        """Check via reasoning model if the context answers the question."""
        prompt = (
            f"Frage: {query}\n\nKontext:\n{context[:1000]}\n\n"  # limit to keep prompt short
            "Reicht der Kontext aus, um die Frage zu beantworten? Antworte nur mit ja oder nein."
        )
        answer = self._think(prompt).strip().lower()
        return answer.startswith("ja")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _run_similarity_search(self, query: str) -> str:
        """Return a context string built from a semantic similarity search."""
        print(f"[DEBUG] Führe similarity_search mit Anfrage: '{query}' aus")
        results = do_a_sim_search(query, k=self.k, vector_store=self.vector_store)
        parts = [f"[{i}] {doc.page_content}" for i, doc in enumerate(results, start=1)]
        context = "\n\n--- DOC SEP ---\n\n".join(parts)
        print("[DEBUG] Gefundener Kontext aus similarity_search:\n", context)
        return context

    def _retrieve_files(self, documents: Optional[List[str]] = None) -> str:
        """Load full documents and return a combined context string."""
        available = list_docx_files()
        if documents:
            docs = [d for d in available if d.name in documents]
        else:
            docs = available
        print("[DEBUG] Lade folgende Dokumente:", [d.name for d in docs])
        context, _ = build_compare_context(docs, self.embedding_model_name)
        print("[DEBUG] Zusammengesetzter Kontext aus Dateien:\n", context)
        return context

    def _generate_final_answer(self, query: str, context: str) -> str:
        """Generate the final answer conditioned on the context."""
        llm = OllamaLLM(model=self.model, num_ctx=16384)
        prompt = PROMPT_TEMPLATE.format(context=context, query=query)
        result = llm.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def run(self, query: str) -> str:
        """Execute the ReAct‑CoT‑SC loop for a user query."""
        # We allow up to two iterations
        context = ""
        for step in range(2):
            print(f"\n[DEBUG] Iteration {step + 1}: Starte Entscheidungsfindung")
            decision = self._decide_action(query)
            print("[DEBUG] Gewählte Aktion:", decision.action, "| Anfrage:", decision.query)
            if decision.action == "similarity_search":
                context = self._run_similarity_search(decision.query)
            else:
                context = self._retrieve_files()
            sufficient = self._is_context_sufficient(query, context)
            print("[DEBUG] Kontext ausreichend?", sufficient)
            if sufficient:
                break
        if not context:
            return "[INFO] Es konnte kein Kontext erstellt werden."
        return self._generate_final_answer(query, context)
