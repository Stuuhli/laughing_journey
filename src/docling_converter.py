from docling.document_converter import DocumentConverter
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_post_process import correct_markdown_numbering
from utils import DOCX_DIR, CONVERTED_DIR


def convert_all_docx(update=False, doc_path=None):
    """
    Converts all .docx files to Markdown. Optional: Single document or update mode.
    """
    CONVERTED_DIR.mkdir(exist_ok=True)

    if doc_path:
        source_files = [doc_path]
    else:
        source_files = list(DOCX_DIR.glob("*.docx"))

    if not source_files:
        print(f"[ERROR] Cloud not find .docx files in given directory: {DOCX_DIR}")
        return

    print(f"[INFO] {len(source_files)} .docx files found. Beginning conversion and correction...")

    converter = DocumentConverter()

    for file_path in source_files:
        try:
            print(f"[INFO] Processing: '{file_path}' ...")
            # Step 1: convert
            doc = converter.convert(file_path).document
            # Step 2: serialize to md
            serializer = MarkdownDocSerializer(doc=doc)
            serialized_text = serializer.serialize().text
            # Step 3: define output file and save
            import os
            output_filename = os.path.splitext(os.path.basename(str(file_path)))[0] + ".txt"
            final_CONVERTED_PATH = CONVERTED_DIR / output_filename
            if final_CONVERTED_PATH.exists():
                print(f"[INFO] Delete existing version: {final_CONVERTED_PATH}")
                final_CONVERTED_PATH.unlink()
            final_CONVERTED_PATH.write_text(serialized_text, encoding="utf-8")
            # Step 4: Call post-processing function from docling_cleaner.py
            correct_markdown_numbering(final_CONVERTED_PATH)
        except Exception as e:
            print(f"[ERROR] Could not process {file_path}: {e}")

    print(f"\n[SUCCESS] Results saved in '{CONVERTED_DIR}'.")


if __name__ == "__main__":
    convert_all_docx()
