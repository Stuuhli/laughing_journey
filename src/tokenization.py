"""
Es hat sich beim Chunking gezeigt, dass manchmal die Chunks eine Länge von 514 statt 512 Token aufweisen.
Das kommt daher, dass BERT und SentenceTransformer [CLS] und [SEP] Token als Begrenzer einfügt.
Durch die Truncation würden die letzten Wörter des Chunks, sowie der SEP Token abgeschnitten werden. Das ist unschön.
Besser wäre eine gesonderte Tokenization auf 510, sodass mit den Begrenzern die 512 Token genau ausgenutzt werden.
"""

from transformers import AutoTokenizer


def tokenize_and_chunk(text, model_name: str, max_context_tokens: int):
    """
    Chunkt Text so, dass inklusive [CLS]/[SEP] nie das Modell-Limit überschritten wird.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_tokens = max_context_tokens - 2  # Reserve für CLS und SEP

    # Tokenisiere OHNE Special Tokens!
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks
