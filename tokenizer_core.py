# tokenizer_core.py - resilient token extraction for Java source
# uses javalang when possible; falls back to a light-weight regex tokenizer on failure

import re
from typing import List

def _simple_tokenize_text(text: str) -> List[str]:
    # Remove string literals first (simple heuristic)
    text = re.sub(r'\"(\\\\.|[^"\\\\])*\"', ' ', text)
    text = re.sub(r"\'(\\\\.|[^'\\\\])*\'", ' ', text)
    # Remove single-line comments
    text = re.sub(r'//.*', ' ', text)
    # Remove block comments roughly (non-greedy)
    text = re.sub(r'/\\*.*?\\*/', ' ', text, flags=re.S)
    # Split on non-word characters (keep identifiers, numbers)
    tokens = re.split(r'\\W+', text)
    tokens = [t for t in tokens if t and not t.isdigit()]
    return tokens

try:
    import javalang
    from javalang.tokenizer import LexerError
    def extract_lexical_stream(path: str):
        try:
            with open(path, 'r', errors='ignore') as fh:
                text = fh.read()
            # javalang expects source input; tokenization over full text works
            tokens = [tok.value for tok in javalang.tokenizer.tokenize(text)]
            return tokens
        except (LexerError, Exception):
            # fallback
            return _simple_tokenize_text(text)
except Exception:
    # javalang not available; use regex fallback only
    def extract_lexical_stream(path: str):
        try:
            with open(path, 'r', errors='ignore') as fh:
                text = fh.read()
        except Exception:
            return []
        return _simple_tokenize_text(text)
