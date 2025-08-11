import re
from pathlib import Path


def reduce_multiple_spaces(text: str) -> str:
    """Reduce sequences of ten or more spaces to exactly ten.

    Args:
        text (str): Input text.

    Returns:
        str: Text with collapsed spaces.
    """
    # Regex to find 10 or more spaces and replace with 10 spaces
    return re.sub(r' {10,}', '          ', text)


def correct_markdown_numbering(file_path: Path):
    """Fix chapter numbering in a markdown file.

    Args:
        file_path (Path): Path to the file to be corrected.

    Returns:
        None
    """
    try:
        original_text = file_path.read_text(encoding="utf-8")
        cleaned_text = reduce_multiple_spaces(original_text)
        lines = cleaned_text.splitlines()

        corrected_lines = []
        toc_found_and_fixed = False

        for line in lines:
            # regex101.com
            match = re.match(r"^(#*\s+)(\d+)(.*)", line)

            if not match:
                corrected_lines.append(line)
                continue

            prefix, number_str, rest = match.groups()

            # has to manually be updated in case more strings are discovered
            # remove numbering from table of contents
            if not toc_found_and_fixed and ("inhaltsverzeichnis" in rest.lower() or "table of contents" in rest.lower()):
                corrected_lines.append(f"##{rest.strip()}")
                toc_found_and_fixed = True

            # In case the toc is already correct (ex. different word template where toc is not defined as heading)
            elif toc_found_and_fixed:
                new_number = int(number_str) - 1
                if new_number > 0:
                    corrected_lines.append(f"{prefix}{new_number}{rest}")
                else:
                    corrected_lines.append(line)
            else:
                corrected_lines.append(line)

        corrected_text = "\n".join(corrected_lines)

        # Only serialize if any changes were made
        # Prevents unnecessary file operations and helps with debugging: no change -> no console log
        if corrected_text != original_text:
            file_path.write_text(corrected_text, encoding="utf-8")
            print(f"    -> [SUCCESS] Chapters corrected in '{file_path.name}'.")

    except FileNotFoundError:
        print(f"    -> [ERROR] File not found in {file_path}")
    except Exception as e:
        print(f"    -> [ERROR] Exception occured during processing of {file_path.name}: {e}")
