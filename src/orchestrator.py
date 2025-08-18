"""
Zunächst soll die query durch ein thinking-Modell analysiert werden.
Dieses Modell kann die Anfrage umformulieren und basierend auf dem Wissen der verfügbaren Dokumente an eine Unterfunktion weitergeben.

Der Aufbau orientiert sich an ReAct mit CoT (ggfs. mit SC, Self-Consistency)

Zentral ist immer das thinking-Modell, dass orchestriert, was getan werden soll. Dafür muss es sich auch daran erinnern können, was es bereits getan hat.
Dieses Modell gibt auch am Ende das "GO" und ruft eine Funktion zum formulieren der Antwort auf, wobei es dort einen eigenen Prompt mitgibt, bzw. den vordefinierten Prompt anpasst.


Ein Beispielablauf könnte also so aussehen:

Anfrage: Deckt mein Dokument zur Leitlinie die Anforderungen der ISO27001 ab?
Thinking-LLM: Der Benutzer fragt nach nach Informationen zu der IS_Leitlinie und der ISO27001. Ich habe Zugriff auf Verzeichnisse, die "IS-Dokumente" und "Gesetze und Normen" heißen. Ich suche in beiden Verzeichnissen nach relevanten Informationen:
Act: Führe similarity-search durch und lasse mit k=20 jeweils Leitlinie und ISO27001 zurückgeben.
Thinking-LLM: Die Ergebnisse sind nicht ausreichend. Ich ziehe mir stattdessen das gesamte Dokument.
Act: Rufe jeweils die gesamten Dokumente in den Kontext.
Thinking-LLM: Die Informationen sind nun ausreichend, um die Anfrage zu beantworten. Weitergabe an Antwortgenerator.
Act: Lade beide Dokumente in den Kontext von z.B. mistral-nemo:12b und gebe originale Nutzeranfrage bei.


Ob die Thinking-LLM (ich könnte die orchestrator nennen) wirklich Zustände braucht, weiß ich noch nicht. Im von mir erfundenen Beispiel würde eine erinnerungslose LLM vermutlich reichen.


Wenn explizit im Dokument steht: "Für weitere Informationen, siehe XY", dann muss diese Information als solche extrahiert werden.
fact: string; //  Provide information directly relevant to the question (or where to find more information in the text) - either supplementary data, facts, or where the answer might be located, like pages and sections.
    Add definitions and other context from the page into the fact, so it's self-explanatory.
relevance: string; // How is this fact relevant to the answer?
nextSource: string; // a page number, a section name, or other descriptors of where to look for more information.
expectedInfo: string; // What information do you expect to find there?


intent → geeignete Tools aufrufen → Evidenz prüfen → ggf. nachziehen → finalisieren
"""


# orchestrator.py
from __future__ import annotations
import json
import textwrap
from typing import Any, Dict, List, Tuple
from pathlib import Path

from langchain_ollama import OllamaLLM

from vector_db import get_vector_store, do_a_sim_search
from utils import (
    get_chunk_dir_for_model,
    load_chunks,
    ensure_converted_text_for_docx,
    load_chunks_for_source,
    DOCX_DIR
)
from answer_generation import generate_answer, generate_answer_compare_docs


# -------------------------------
# Tool-Implementierungen (Python)
# -------------------------------

def tool_search_chunks(query: str, k: int, embedding_model: str) -> Dict[str, Any]:
    vs = get_vector_store(embedding_model_name=embedding_model)
    docs = do_a_sim_search(query=query, k=k, vector_store=vs)
    hits = []
    for d in docs:
        hits.append({
            "content": d.page_content,
            "meta": dict(d.metadata) if hasattr(d, "metadata") else {},
        })
    return {"hits": hits, "k": k, "embedding_model": embedding_model}


def tool_expand_to_full_chapters(hits: Dict[str, Any], embedding_model: str) -> Dict[str, Any]:
    # Sammle Kapitel aus Hits
    chapter_ids = {
        h["meta"].get("chapter_path")
        for h in hits.get("hits", [])
        if h.get("meta") and h["meta"].get("chapter_path")
    }
    # Alle Chunks dieses Modells laden und Kapitel-Blocks bauen
    all_items = load_chunks(chunk_dir=get_chunk_dir_for_model(embedding_model))
    grouped: Dict[Tuple[str, str], List[str]] = {}
    for it in all_items:
        meta = it.get("metadata", {})
        chap = meta.get("chapter_path")
        if chap in chapter_ids:
            key = (meta.get("source_file", "Unknown file"), chap)
            grouped.setdefault(key, []).append(it.get("page_content", ""))

    parts, sources = [], []
    for i, key in enumerate(grouped.keys(), start=1):
        parts.append(f"[{i}] " + "\n".join(grouped[key]))
        sources.append({"id": i, "source_file": key[0], "chapter_path": key[1]})

    return {"context": "\n\n--- DOC SEP ---\n\n".join(parts), "sources": sources}


def tool_list_docs() -> Dict[str, Any]:
    files = list(DOCX_DIR.glob("*.docx"))
    return {
        "docx_paths": [str(p) for p in files],
        "stems": [p.stem for p in files],
    }


def tool_load_full_docs(docx_paths: List[str], embedding_model: str) -> Dict[str, Any]:
    parts, srcs = [], []
    path_list = _resolve_docx_paths(docx_paths)
    for i, p in enumerate(path_list, start=1):
        import os
        source_txt_filename = os.path.splitext(p.name)[0] + ".txt"
        chunk_texts = load_chunks_for_source(embedding_model, source_txt_filename)
        if chunk_texts:
            parts.append(f"[{i}] " + "\n".join(chunk_texts))
            srcs.append({"id": i, "label": f"{source_txt_filename} (Chunks)"})
        else:
            txt_path = ensure_converted_text_for_docx(p)
            parts.append(f"[{i}] " + txt_path.read_text(encoding="utf-8"))
            srcs.append({"id": i, "label": f"{source_txt_filename} (Converted TXT)"})
    return {"context": "\n\n--- DOC SEP ---\n\n".join(parts), "sources": srcs}


def tool_final_answer_compare(model: str, query: str, docx_paths: List[str], embedding_model: str) -> str:
    path_objs = _resolve_docx_paths(docx_paths)
    return generate_answer_compare_docs(
        model=model,
        query=query,
        selected_docx_paths=path_objs,
        embedding_model_name=embedding_model,
    )


def tool_final_answer_rag(model: str, query: str, k: int, embedding_model: str, full_chapters: bool) -> str:
    vs = get_vector_store(embedding_model_name=embedding_model)
    return generate_answer(
        model=model,
        query=query,
        k_values=k,
        vector_store=vs,
        embedding_model_name=embedding_model,
        use_full_chapters=full_chapters,
    )


# -------------------------------
# Orchestrator (Planner-Loop)
# -------------------------------

SYSTEM_INSTRUCTIONS = """Du bist ein Planner/Controller für ein Agentic-RAG-System.
Arbeite im ReAct-Stil mit kurzen Schritten:
- Formuliere den nächsten Schritt als JSON.
- Nutze so wenige Schritte wie möglich.
- Wenn Vergleich zweier Dokumente sinnvoll ist, nutze den Vergleichspfad.
- Wenn du final_answer_compare verwenden willst, rufe ZUERST list_docs auf und
  übergib bei final_answer_compare AUSSCHLIESSLICH exakte Pfade aus list_docs.docx_paths.
Gib als Ausgabe NUR genau ein JSON-Objekt mit diesem Schema zurück:

{
  "tool": "<NAME>",
  "args": { ... }
}

Verfügbare Tools und Argumente:
- search_chunks: { "query": str, "k": int, "embedding_model": str }
- expand_to_full_chapters: { }  // erweitert die LETZTEN Treffer (aus search_chunks)
- list_docs: { }
- load_full_docs: { "docx_paths": [str], "embedding_model": str }
- final_answer_rag: { "model": str, "query": str, "k": int, "embedding_model": str, "full_chapters": bool }
- final_answer_compare: { "model": str, "query": str, "docx_paths": [str], "embedding_model": str }

Wähle am Ende IMMER eines der final_* Tools.
"""


def _as_text(x: Any) -> str:
    return x.content if hasattr(x, "content") else str(x)


def _build_planner_prompt(scratchpad: str, user_query: str, embedding_model: str, gen_model: str, k: int) -> str:
    return textwrap.dedent(f"""
    {SYSTEM_INSTRUCTIONS}

    Kontext:
    - EmbeddingModel: {embedding_model}
    - Default k: {k}
    - GenModel: {gen_model}

    Verlauf (Thought/Action/Observation):
    {scratchpad.strip() if scratchpad else "(leer)"}

    Aufgabe:
    - Nutzerfrage: {user_query}
    - Erzeuge NUR EIN JSON-Objekt (kein Text davor/danach).
    """)


def _resolve_docx_paths(docx_paths: List[str]) -> List[Path]:
    """
    Normalisiert vom Planner kommende Strings wie 'Passwortrichtlinie' zu echten .docx-Pfaden.
    Strategie:
      1) Akzeptiere absolute/relative Pfade, falls existent.
      2) Falls nur Name/Stichwort: suche im DOCX_DIR nach passendem Stem (exact, dann substring).
    """
    catalog = list(DOCX_DIR.glob("*.docx"))
    by_stem = {p.stem.lower(): p for p in catalog}
    resolved: List[Path] = []

    for entry in docx_paths:
        s = str(entry).strip().strip('"').strip("'")
        cand = Path(s)

        # 1) existierender Pfad (mit/ohne .docx)
        if cand.exists():
            if cand.suffix.lower() == ".docx":
                resolved.append(cand)
            else:
                # Falls Ordner oder falsche Endung – versuche .docx daneben
                docx_try = cand.with_suffix(".docx")
                if docx_try.exists():
                    resolved.append(docx_try)
                else:
                    raise FileNotFoundError(f"Pfad existiert, aber keine .docx gefunden: {cand}")
            continue

        # 2) Im DOCX_DIR mit exakter Datei
        if cand.suffix.lower() == ".docx":
            p = DOCX_DIR / cand.name
            if p.exists():
                resolved.append(p)
                continue

        # 3) Stem-Match im DOCX_DIR (exakt, case-insensitiv)
        name = (cand.stem if cand.suffix else cand.name).lower()
        if name in by_stem:
            resolved.append(by_stem[name])
            continue

        # 4) Substring-Suche im Stem
        subs = [p for p in catalog if name in p.stem.lower()]
        if subs:
            # nimm das erste – optional: Ranking einbauen
            resolved.append(subs[0])
            continue

        raise FileNotFoundError(f"Kein DOCX zu '{entry}' in {DOCX_DIR} gefunden.")

    return resolved


def _parse_json(s: str) -> Dict[str, Any] | None:
    s = s.strip()
    # Versuche direktes JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # Suche erstes {...} per Heuristik
    import re
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def run_orchestrator(user_query: str, embedding_model: str, gen_model: str, k: int = 10) -> str:
    """
    Planner-Loop ohne bind_tools. Lässt jedes Modell laufen, das einfache Textgenerierung kann.
    """
    llm = OllamaLLM(model=gen_model, num_ctx=16384)

    scratchpad = ""
    last_hits: Dict[str, Any] | None = None

    MAX_STEPS = 4
    for step in range(1, MAX_STEPS + 1):
        prompt = _build_planner_prompt(scratchpad, user_query, embedding_model, gen_model, k)
        raw = llm.invoke(prompt)
        decision = _parse_json(_as_text(raw))
        if not decision or "tool" not in decision:
            # Fallback: direkter Ein-Schritt-RAG
            return tool_final_answer_rag(
                model=gen_model, query=user_query, k=k,
                embedding_model=embedding_model, full_chapters=False
            )

        name = decision["tool"]
        args = decision.get("args", {}) or {}

        # Ausführen & Observation
        if name == "search_chunks":
            res = tool_search_chunks(
                query=args.get("query", user_query),
                k=int(args.get("k", k)),
                embedding_model=args.get("embedding_model", embedding_model),
            )
            last_hits = res
            obs_short = f"hits={len(res.get('hits', []))}"
            scratchpad += f"\nThought: Suche nach relevanten Chunks.\nAction: search_chunks\nObservation: {obs_short}\n"

        elif name == "expand_to_full_chapters":
            if not last_hits:
                # Nichts zu expandieren → zurück zur Suche
                scratchpad += "\nThought: Keine Hits vorhanden; Suche erneut.\n"
                continue
            res = tool_expand_to_full_chapters(last_hits, embedding_model=embedding_model)
            # Observation verkürzen
            obs_short = f"context_len={len(res.get('context', ''))}, sources={len(res.get('sources', []))}"
            # Kontext im Scratchpad nur kurz referenzieren
            scratchpad += f"\nThought: Kontext je Kapitel aggregiert.\nAction: expand_to_full_chapters\nObservation: {obs_short}\n"
            # Speichere „synthetische“ Hits als Kontext-Placeholder
            last_hits = {"hits": [], "context": res.get("context", ""), "sources": res.get("sources", [])}

        elif name == "list_docs":
            res = tool_list_docs()
            obs_short = f"docs={len(res.get('docx_paths', []))}"
            scratchpad += f"\nThought: Prüfe verfügbare DOCX.\nAction: list_docs\nObservation: {obs_short}\n"

        elif name == "load_full_docs":
            res = tool_load_full_docs(
                docx_paths=list(args.get("docx_paths", [])),
                embedding_model=args.get("embedding_model", embedding_model),
            )
            obs_short = f"context_len={len(res.get('context', ''))}, sources={len(res.get('sources', []))}"
            scratchpad += f"\nThought: Lade vollständige Dokumente.\nAction: load_full_docs\nObservation: {obs_short}\n"
            # Ablegen für finalen Schritt
            last_hits = {"hits": [], "context": res.get("context", ""), "sources": res.get("sources", [])}

        elif name == "final_answer_rag":
            # Wenn vorher expand_to_full_chapters gelaufen ist, können wir full_chapters=True setzen
            full_chapters = bool(args.get("full_chapters", bool(last_hits and last_hits.get("sources"))))
            return tool_final_answer_rag(
                model=args.get("model", gen_model),
                query=args.get("query", user_query),
                k=int(args.get("k", k)),
                embedding_model=args.get("embedding_model", embedding_model),
                full_chapters=full_chapters,
            )

        elif name == "final_answer_compare":
            return tool_final_answer_compare(
                model=args.get("model", gen_model),
                query=args.get("query", user_query),
                docx_paths=list(args.get("docx_paths", [])),
                embedding_model=args.get("embedding_model", embedding_model),
            )

        else:
            # Unbekanntes Tool → Fallback
            return tool_final_answer_rag(
                model=gen_model, query=user_query, k=k,
                embedding_model=embedding_model, full_chapters=False
            )

    # Harte Step-Grenze erreicht → Fallback
    return tool_final_answer_rag(
        model=gen_model, query=user_query, k=k,
        embedding_model=embedding_model, full_chapters=False
    )
