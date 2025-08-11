from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
import json
import re
import hashlib
from utils import sliding_window_chunk, CHUNKS_DIR

PROJECT_DIR = Path(__file__).parent.parent
INPUT_PATH = PROJECT_DIR / "data" / "converted"
OUTPUT_PATH = PROJECT_DIR / "data" / "chunks"

HEADER_TYPES = [("##", "Chapter"), ("###", "Sub-Chapter"), ("####", "Sub-Sub-Chapter")]


def parse_chapter(header: str):
    """Extract chapter number and title, e.g. “3.1 Access control” → (“3.1”, “Access control”)"""
    if not header:
        return None, None
    m = re.match(r"^([\d.]+)\s*(.*)$", header.strip())
    if m:
        return m.group(1), m.group(2)
    return None, header.strip()


def chunk_md_headers(text_content: str):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADER_TYPES, strip_headers=False
    )
    return splitter.split_text(text=text_content)


def get_chunk_id(text: str, meta: dict) -> str:
    s = json.dumps(meta, sort_keys=True, ensure_ascii=False) + text
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def create_chunks_for_all(update=False, doc_path=None, max_chunk_length=1024, chunk_dir=None):
    """
    Creates chunks for all (or a single) converted document(s). Optional: update mode.
    """
    # the max_chunk_length has to me multiplied by 3.5 to reach the effect of chunk -> token conversion, which is a factor between roughly 4-6
    max_chunk_length = int(max_chunk_length * 3.5)
    if chunk_dir is None:
        chunk_dir = CHUNKS_DIR
    chunk_dir.mkdir(exist_ok=True)
    OUTPUT_PATH.mkdir(exist_ok=True)
    if doc_path:
        input_files = [Path(doc_path)]
    else:
        if not INPUT_PATH.is_dir():
            print(f"[ERROR] Could not find input directory: {INPUT_PATH}")
            return
        input_files = list(INPUT_PATH.glob("*.txt"))

    files_processed = 0

    for file_path in input_files:
        print(f"[INFO] Processing file: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        chunks = chunk_md_headers(text_content=file_content)
        filtered_chunks = []

        for chunk in chunks:
            meta = chunk.metadata.copy()
            meta["source_file"] = file_path.name

            # Kapitelnummer und Titel pro Ebene extrahieren, wenn vorhanden
            chapter_titles = []
            for key in ["Chapter", "Sub-Chapter", "Sub-Sub-Chapter"]:
                if key in meta and meta[key]:
                    num, title = parse_chapter(meta[key])
                    if num:
                        meta[f"{key}_number"] = num
                    if title:
                        meta[f"{key}_title"] = title
                    chapter_titles.append(f"{num} {title}" if num else title)

            # Kapitelstruktur als String abbilden
            if chapter_titles:
                meta["chapter_path"] = " > ".join(chapter_titles)

            # Leere Chunks filtern
            content = chunk.page_content.strip()
            if not content:
                continue

            if len(content) > max_chunk_length:
                subchunks = sliding_window_chunk(content, max_length=max_chunk_length)
                for i, sub in enumerate(subchunks):
                    sub_meta = meta.copy()
                    sub_meta["subchunk"] = i
                    sub_meta["chunk_id"] = get_chunk_id(sub, sub_meta)
                    filtered_chunks.append({"page_content": sub, "metadata": sub_meta})
            else:
                meta["chunk_id"] = get_chunk_id(content, meta)
                filtered_chunks.append({"page_content": content, "metadata": meta})

        if not filtered_chunks:
            print(f"[WARN] No chunks created for file: {file_path.name}")
            continue

        output_json_path = chunk_dir / f"{file_path.stem}.json"
        if update and output_json_path.exists():
            print(f"[INFO] Overwrite existing file: {output_json_path}")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(filtered_chunks, f, ensure_ascii=False, indent=4)

        files_processed += 1

    print("-" * 20)
    print(f"[SUCCESS] Finished processing. {files_processed} JSON files created in {OUTPUT_PATH}")
