"""
Dataset Preprocessing, Cleaning, Deduplication & Quality Filtering for URA-Shree.
Processes raw source code, technical documentation, and agent interaction dialogues
while strictly preserving code indentation and structural semantics.
"""

import re
import hashlib
import unicodedata
from typing import List, Dict, Set, Optional, Tuple, Iterable


# Permissive open-source licenses compatible with model training
PERMISSIVE_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0"}


def clean_text(text: str) -> str:
    """
    Cleans raw text while strictly preserving indentation (spaces and tabs).
    - Normalizes Unicode to NFC.
    - Strips carriage returns (\\r\\n -> \\n).
    - Removes non-printable control characters except \\n and \\t.
    - Collapses 3+ consecutive newlines into 2 (paragraph break).
    """
    if not text:
        return ""

    # Normalize Unicode representations
    text = unicodedata.normalize("NFC", text)

    # Standardize newline characters
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Filter control characters: retain printable chars, newlines, and tabs
    cleaned_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in ("\n", "\t") or not cat.startswith("C"):
            cleaned_chars.append(ch)
        else:
            # Replace control char with single space
            cleaned_chars.append(" ")
    text = "".join(cleaned_chars)

    # Collapse excessive blank lines (more than 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def compute_content_hash(text: str, normalize_whitespace: bool = False) -> str:
    """
    Computes a cryptographic SHA-256 fingerprint of the text for deduplication.
    If normalize_whitespace is True, ignores minor whitespace differences (near-dedup).
    """
    if normalize_whitespace:
        # Collapse all whitespace sequences into single spaces for near-deduplication
        normalized = " ".join(text.split()).strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DataFilter:
    """
    Quality and heuristic filter to discard low-quality, minified, or corrupt code/text.
    """
    def __init__(
        self,
        min_chars: int = 40,
        max_chars: int = 500_000,
        min_alpha_ratio: float = 0.25,
        max_line_length: int = 1500,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_alpha_ratio = min_alpha_ratio
        self.max_line_length = max_line_length

    def is_valid(self, text: str) -> Tuple[bool, str]:
        """
        Evaluates whether a text sample meets quality standards.
        Returns (is_valid, reason).
        """
        length = len(text)
        if length < self.min_chars:
            return False, f"Too short ({length} < {self.min_chars} chars)"
        if length > self.max_chars:
            return False, f"Too long ({length} > {self.max_chars} chars)"

        # Check for minified code / single gigantic lines
        lines = text.split("\n")
        for line in lines:
            if len(line) > self.max_line_length:
                return False, f"Line exceeds maximum length ({len(line)} > {self.max_line_length})"

        # Check alphanumeric content ratio
        alpha_count = sum(1 for c in text if c.isalnum())
        alpha_ratio = alpha_count / max(1, length)
        if alpha_ratio < self.min_alpha_ratio:
            return False, f"Alphanumeric ratio too low ({alpha_ratio:.2f} < {self.min_alpha_ratio})"

        return True, "valid"


def deduplicate_corpus(
    corpus: List[str],
    normalize_whitespace: bool = False
) -> List[str]:
    """
    Removes exact or near-duplicate documents from a list of texts using SHA-256 hashing.
    Preserves first-seen order.
    """
    seen_hashes: Set[str] = set()
    unique_corpus: List[str] = []

    for doc in corpus:
        doc_hash = compute_content_hash(doc, normalize_whitespace=normalize_whitespace)
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_corpus.append(doc)

    return unique_corpus


def format_agent_document(
    system_prompt: str,
    user_prompt: str,
    assistant_response: str,
    tool_calls: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Formats an agent interaction with appropriate boundary tokens for SFT and pretraining.
    """
    doc = f"<|bos|><|system|>\n{system_prompt.strip()}\n"
    doc += f"<|user|>\n{user_prompt.strip()}\n"
    doc += f"<|assistant|>\n"

    if tool_calls:
        for call in tool_calls:
            doc += f"<|tool_call|>\n{call.get('invocation', '').strip()}\n"
            if "result" in call:
                doc += f"<|tool_result|>\n{call.get('result', '').strip()}\n"

    doc += f"{assistant_response.strip()}\n<|eos|>"
    return doc
