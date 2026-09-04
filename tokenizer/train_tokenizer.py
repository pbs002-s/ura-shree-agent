"""
BPE Tokenizer Training Script for URA-Shree.
Learns byte-pair merge rules iteratively from a corpus of text and source code.
No external tokenizer libraries used.
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.vocabulary import Vocabulary, SPECIAL_TOKENS, bytes_to_unicode
from tokenizer.tokenizer import BPETokenizer


def get_pair_frequencies(
    word_freqs: Dict[Tuple[str, ...], int]
) -> Dict[Tuple[str, str], int]:
    """Frequency of every adjacent symbol pair, weighted by word frequency."""
    pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_freqs[(word[i], word[i + 1])] += freq
    return pair_freqs


def merge_word_pair(
    pair: Tuple[str, str], word_freqs: Dict[Tuple[str, ...], int]
) -> Dict[Tuple[str, ...], int]:
    """Replaces every occurrence of `pair` across all words with the merged symbol."""
    first, second = pair
    merged = first + second
    result: Dict[Tuple[str, ...], int] = {}

    for word, freq in word_freqs.items():
        result[_merge_one(word, first, second, merged)] = freq
    return result


def _merge_one(
    word: Tuple[str, ...], first: str, second: str, merged: str
) -> Tuple[str, ...]:
    """Applies one merge to a single word."""
    out: List[str] = []
    i = 0
    limit = len(word) - 1
    while i < len(word):
        if i < limit and word[i] == first and word[i + 1] == second:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_bpe(
    corpus: List[str],
    target_vocab_size: int = 1000,
    special_tokens: Optional[List[str]] = None,
    verbose: bool = True,
    min_pair_frequency: int = 2,
) -> BPETokenizer:
    """
    Trains a byte-level BPE tokenizer.

    The merge loop is incremental. Recomputing every pair frequency after each
    merge is O(merges * corpus), which is fine for a toy corpus and unusable on
    a real one - learning 3500 merges over a million characters would take tens
    of minutes. Here each merge updates only the words that actually contained
    the merged pair, and adjusts the affected pair counts by hand, which makes
    the cost proportional to what changed rather than to the whole corpus.

    Args:
        corpus: source documents
        target_vocab_size: total vocabulary including specials and the 256 bytes
        special_tokens: control tokens to register first
        verbose: print progress
        min_pair_frequency: stop once the best remaining pair is rarer than this
    """
    if special_tokens is None:
        special_tokens = list(SPECIAL_TOKENS)

    vocab = Vocabulary()
    for tok in special_tokens:
        vocab.add_token(tok)

    byte_encoder = bytes_to_unicode()
    for b in range(256):
        vocab.add_token(byte_encoder[b])

    base_size = len(vocab)
    num_merges_needed = max(0, target_vocab_size - base_size)

    if verbose:
        print(f"[BPE Trainer] Base vocabulary: {base_size} tokens "
              f"({len(special_tokens)} special + 256 bytes).")
        print(f"[BPE Trainer] Target {target_vocab_size}, so {num_merges_needed} merges to learn.")

    # Pre-tokenize into byte-level symbol tuples, counted by frequency.
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    for text in corpus:
        for token in BPETokenizer.PAT.findall(text):
            counts[tuple(byte_encoder[b] for b in token.encode("utf-8"))] += 1

    # Indexable parallel arrays: words[i] is a symbol tuple, freqs[i] its count.
    words: List[Tuple[str, ...]] = list(counts)
    freqs: List[int] = [counts[w] for w in words]

    if verbose:
        print(f"[BPE Trainer] {len(words):,} unique word stems from "
              f"{sum(freqs):,} total occurrences.")

    # pair -> total frequency, and pair -> which words contain it. The second
    # index is what makes a merge touch only the affected words.
    pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_words: Dict[Tuple[str, str], set] = defaultdict(set)

    def add_word_pairs(index: int, sign: int) -> None:
        word = words[index]
        freq = freqs[index] * sign
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] += freq
            if sign > 0:
                pair_words[pair].add(index)

    for index in range(len(words)):
        add_word_pairs(index, 1)

    merges: List[Tuple[str, str]] = []

    for step in range(num_merges_needed):
        best_pair, best_freq = None, 0
        for pair, freq in pair_freqs.items():
            if freq > best_freq:
                best_pair, best_freq = pair, freq

        if best_pair is None or best_freq < min_pair_frequency:
            if verbose:
                print(f"[BPE Trainer] Stopping at step {step}: "
                      f"best remaining pair occurs {best_freq} time(s).")
            break

        merges.append(best_pair)
        merged_symbol = best_pair[0] + best_pair[1]
        vocab.add_token(merged_symbol)

        first, second = best_pair
        affected = list(pair_words.get(best_pair, ()))

        for index in affected:
            word = words[index]
            if first not in word:
                continue
            rebuilt = _merge_one(word, first, second, merged_symbol)
            if rebuilt == word:
                continue
            # Retract the old pair counts, install the word, add the new ones.
            add_word_pairs(index, -1)
            words[index] = rebuilt
            add_word_pairs(index, 1)

        # Counts can reach zero; leaving them in makes the scan slower each step.
        pair_freqs.pop(best_pair, None)
        pair_words.pop(best_pair, None)
        for pair in [p for p, f in pair_freqs.items() if f <= 0]:
            pair_freqs.pop(pair, None)
            pair_words.pop(pair, None)

        if verbose and ((step + 1) % 250 == 0 or step == num_merges_needed - 1):
            print(f"[BPE Trainer] {step + 1}/{num_merges_needed} | "
                  f"{best_pair} -> '{merged_symbol}' (freq {best_freq}) | vocab {len(vocab)}")

    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    if verbose:
        print(f"[BPE Trainer] Done. Final vocabulary size: {len(tokenizer)}")
    return tokenizer


# Standard starter training corpus covering programming languages, technical terms, and agent syntax
SAMPLE_TRAINING_CORPUS = [
    """
    def binary_search(arr, target):
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1
    """,
    """
    class Transformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
            self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
            self.norm = LayerNorm(config.embed_dim)
            self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        def forward(self, idx):
            x = self.embed(idx)
            for layer in self.layers:
                x = layer(x)
            return self.head(self.norm(x))
    """,
    """
    import os
    import sys
    import json
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def compute_loss(logits, targets):
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return loss
    """,
    """
    <|system|>
    You are Shree, an autonomous AI coding assistant.
    You inspect codebases, execute terminal commands safely, write tests, and fix bugs.
    <|user|>
    Create a REST API endpoint in FastAPI that returns server health.
    <|assistant|>
    <|tool_call|>
    {"name": "filesystem.write", "path": "app.py", "content": "from fastapi import FastAPI\\napp = FastAPI()\\n\\n@app.get('/health')\\ndef health():\\n    return {'status': 'ok'}\\n"}
    <|tool_result|>
    {"success": true, "bytes_written": 128}
    The health check endpoint is implemented.
    """,
    """
    function calculateFibonacci(n: number): number {
        if (n <= 1) return n;
        let prev = 0, curr = 1;
        for (let i = 2; i <= n; i++) {
            const next = prev + curr;
            prev = curr;
            curr = next;
        }
        return curr;
    }
    export default calculateFibonacci;
    """,
    """
    SELECT users.id, users.username, COUNT(orders.id) as total_orders
    FROM users
    LEFT JOIN orders ON users.id = orders.user_id
    WHERE orders.status = 'completed'
    GROUP BY users.id, users.username
    HAVING COUNT(orders.id) > 5
    ORDER BY total_orders DESC;
    """,
    """
    git status
    git add .
    git commit -m "Implement BPE tokenizer and Transformer model from scratch"
    git push origin main
    npm run build
    pytest tests/ -v
    python training/train.py --config configs/small.yaml
    """
]


def main():
    parser = argparse.ArgumentParser(description="Train the URA-Shree BPE tokenizer.")
    parser.add_argument("--vocab-size", type=int, default=4096,
                        help="Target vocabulary size including 256 byte tokens and specials.")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Train on a single text file instead of the source tree.")
    parser.add_argument("--source-root", type=str, default=".",
                        help="Directory whose source files form the corpus.")
    parser.add_argument("--no-source", action="store_true",
                        help="Use only the small embedded corpus.")
    args = parser.parse_args()

    if args.data_file and os.path.exists(args.data_file):
        with open(args.data_file, "r", encoding="utf-8") as f:
            corpus = [f.read()]
        print(f"[BPE Trainer] Corpus: {args.data_file}")
    elif args.no_source:
        corpus = SAMPLE_TRAINING_CORPUS
        print("[BPE Trainer] Corpus: embedded sample documents.")
    else:
        # Real source files produce far better merges than a handful of
        # hand-written samples: the merges learned are the ones that actually
        # recur in code, so the same text costs fewer tokens.
        from data.source_corpus import collect_documents, corpus_stats

        corpus = collect_documents([args.source_root]) + SAMPLE_TRAINING_CORPUS
        stats = corpus_stats(corpus)
        print(f"[BPE Trainer] Corpus: {stats['documents']} source documents, "
              f"{stats['characters']:,} characters from {args.source_root}")

    tokenizer = train_bpe(corpus, target_vocab_size=args.vocab_size, verbose=True)
    out_path = tokenizer.save(args.output_dir)
    print(f"Tokenizer saved successfully to: {out_path}")


if __name__ == "__main__":
    main()
