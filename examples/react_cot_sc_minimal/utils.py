from pathlib import Path
from typing import List, Tuple

PROMPT_TEMPLATE = """{context}\n\nFrage:\n{query}\n\nAntwort:"""

def list_docx_files() -> List[Path]:
    """Return demo documents shipped with the minimal example."""
    data_dir = Path(__file__).parent / "docs"
    return sorted(data_dir.glob("*.txt"))

def build_compare_context(selected_docx: List[Path], embedding_model_name: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Create a numbered context string from the provided documents."""
    parts = []
    sources: List[Tuple[int, str]] = []
    for i, path in enumerate(selected_docx, start=1):
        text = path.read_text(encoding="utf-8")
        parts.append(f"[{i}] {text}")
        sources.append((i, path.name))
    return "\n\n--- DOC SEP ---\n\n".join(parts), sources
