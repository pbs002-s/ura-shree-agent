"""
URA-Shree Autonomous Distillation and Instruction Fine-Tuning Pipeline.

Uses local Ollama teacher to generate diverse ChatGPT-style conversations,
formats them into Shree's ChatML schema, tokenizes them with Shree's custom BPE,
and trains Shree's Transformer model on GPU with BF16 mixed precision.
Zero external API calls. 100% owned weights.
"""

import os
import sys
import json
import time
import urllib.request
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch

# Ensure UTF-8 console output
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ProjectConfig
from model.model import ShreeTransformerLM
from tokenizer.tokenizer import BPETokenizer
from data.dataset import CausalLMDataset, create_dataloader
from training.evaluate import evaluate_model
from training.train import configure_optimizers, get_lr_cosine_warmup


SEED_PROMPTS = [
    # General Assistant & Conversational
    "Hello! Introduce yourself and explain what you can help me with.",
    "Explain what an API is in simple terms with an everyday analogy.",
    "Give me 3 practical tips for writing clean, maintainable code.",
    "What is the difference between synchronous and asynchronous programming?",
    "Explain how a hash table works and what makes lookups fast.",
    "How do virtual environments work in Python and why should I use them?",
    "Explain what Git branching is and how merge conflicts happen.",
    "What is recursion in computer science? Provide a quick intuitive explanation.",

    # Python Programming
    "Write a Python function to check if a string is a valid palindrome, ignoring spaces and punctuation.",
    "Show me how to read and write JSON files safely in Python with error handling.",
    "Write a Python decorator that measures and prints the execution time of a function.",
    "Explain list comprehensions in Python with 3 clear examples.",
    "Write a Python function to find the longest substring without repeating characters.",
    "How do Python generators work? Show a generator function that yields Fibonacci numbers.",

    # JavaScript / Web Development
    "Explain the difference between let, const, and var in modern JavaScript.",
    "Write a simple JavaScript fetch example with async/await and try/catch error handling.",
    "Explain what React state and props are and how data flows between components.",
    "What is the Virtual DOM and why does React use it?",
    "Write a CSS snippet for centering a div both vertically and horizontally using Flexbox.",

    # Algorithms & Data Structures
    "Implement binary search in Python with comments explaining the logic.",
    "Explain the difference between a stack and a queue with real-world examples.",
    "What is Big-O notation? Compare O(1), O(n), and O(n^2) complexities simply.",
    "Write a Python function to merge two sorted lists into one sorted list.",

    # Debugging & Best Practices
    "Why does '1' + 1 evaluate to '11' in JavaScript while '1' - 1 evaluates to 0?",
    "How can I prevent SQL injection attacks in web applications?",
    "What are HTTP status codes? Explain 200, 400, 401, 404, and 500.",
    "How do I debug a Python script that raises an IndexError: list index out of range?"
]


def query_teacher(prompt: str, model: str = "qwen3.5:0.8b", timeout: int = 40) -> str:
    """Queries local Ollama instance for a high-quality response."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Shree, an intelligent, helpful, and concise AI assistant. "
                    "Provide clear, accurate, and well-structured answers with code examples when relevant."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        content = res["choices"][0]["message"]["content"].strip()
        return content


def format_chat_document(user_prompt: str, assistant_reply: str) -> str:
    """Formats a dialogue into Shree's boundary-tagged ChatML format."""
    doc = (
        "<|bos|><|system|>\n"
        "You are Shree, an autonomous AI language model and intelligent assistant.\n"
        f"<|user|>\n{user_prompt.strip()}\n"
        f"<|assistant|>\n{assistant_reply.strip()}\n<|eos|>"
    )
    return doc


def generate_distillation_corpus(
    teacher_model: str = "qwen3.5:0.8b",
    corpus_cache_path: str = "datasets/distilled_chat.json",
    target_count: int = 25,
) -> List[str]:
    """Generates synthetic multi-turn conversation documents."""
    cache_file = PROJECT_ROOT / corpus_cache_path
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    dialogues: List[Dict[str, str]] = []
    if cache_file.exists():
        try:
            dialogues = json.loads(cache_file.read_text(encoding="utf-8"))
            print(f"[Distill] Loaded {len(dialogues)} cached conversations from {corpus_cache_path}")
        except Exception:
            dialogues = []

    prompts_to_run = SEED_PROMPTS[:target_count]
    existing_prompts = {d["prompt"] for d in dialogues}

    needed = [p for p in prompts_to_run if p not in existing_prompts]
    if needed:
        print(f"[Distill] Generating {len(needed)} new synthetic dialogues from {teacher_model}...")
        for idx, prompt in enumerate(needed, 1):
            try:
                t0 = time.time()
                reply = query_teacher(prompt, model=teacher_model)
                elapsed = time.time() - t0
                dialogues.append({"prompt": prompt, "reply": reply})
                print(f"  [{idx}/{len(needed)}] Generated: \"{prompt[:45]}...\" ({elapsed:.1f}s, {len(reply)} chars)")
            except Exception as e:
                print(f"  [{idx}/{len(needed)}] Failed for prompt: {e}")

        # Cache generated corpus
        cache_file.write_text(json.dumps(dialogues, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Distill] Saved {len(dialogues)} conversations to {corpus_cache_path}")

    # Convert all into ChatML documents
    documents = [format_chat_document(d["prompt"], d["reply"]) for d in dialogues]
    return documents


def prepare_datasets(
    documents: List[str],
    tokenizer_path: str = "checkpoints/tokenizer.json",
    output_dir: str = "datasets",
    val_ratio: float = 0.1,
    replications: int = 40,
) -> Tuple[str, str]:
    """Tokenizes and serializes documents into binary datasets."""
    tok_file = PROJECT_ROOT / tokenizer_path
    if not tok_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_file}")

    tokenizer = BPETokenizer.load(str(tok_file))
    print(f"[Tokenizer] Loaded vocabulary of size {len(tokenizer)}")

    all_tokens: List[int] = []
    # Replicate documents to create adequate training epoch density
    for _ in range(replications):
        for doc in documents:
            tokens = tokenizer.encode(doc)
            all_tokens.extend(tokens)

    total_tokens = len(all_tokens)
    print(f"[Dataset] Total tokens across corpus: {total_tokens:,}")

    split = int(total_tokens * (1.0 - val_ratio))
    train_tokens = all_tokens[:split]
    val_tokens = all_tokens[split:]

    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    train_bin = out_path / "shree_chat_train.bin"
    val_bin = out_path / "shree_chat_val.bin"

    np.array(train_tokens, dtype=np.uint16).tofile(str(train_bin))
    np.array(val_tokens, dtype=np.uint16).tofile(str(val_bin))

    print(f"[Dataset] Written {train_bin.name}: {len(train_tokens):,} tokens ({train_bin.stat().st_size / 1024:.1f} KB)")
    print(f"[Dataset] Written {val_bin.name}: {len(val_tokens):,} tokens ({val_bin.stat().st_size / 1024:.1f} KB)")

    return str(train_bin), str(val_bin)


def train_shree_model(
    config_path: str = "configs/small.yaml",
    base_checkpoint: str = "checkpoints/best.pt",
    output_checkpoint: str = "checkpoints/shree_chat_best.pt",
    train_bin: str = "datasets/shree_chat_train.bin",
    val_bin: str = "datasets/shree_chat_val.bin",
    steps: int = 60,
    learning_rate: float = 2.5e-4,
    batch_size: int = 8,
) -> str:
    """Trains the local Shree Transformer on the prepared chat dataset."""
    print("=" * 72)
    print("      TRAINING URA-SHREE CHAT MODEL (NVIDIA RTX 4060 GPU)")
    print("=" * 72)

    config = ProjectConfig.load_from_yaml(str(PROJECT_ROOT / config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Compute] Target device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model = ShreeTransformerLM(config.model).to(device)

    base_path = PROJECT_ROOT / base_checkpoint
    if base_path.exists():
        print(f"[Model] Initializing from base weights: {base_path.name}")
        raw = torch.load(str(base_path), map_location=device, weights_only=False)
        model.load_state_dict(raw["model_state_dict"])
    else:
        print("[Model] Base checkpoint not found, initializing fresh weights.")

    seq_len = min(512, config.model.max_seq_len)
    train_ds = CausalLMDataset(train_bin, seq_len=seq_len)
    val_ds = CausalLMDataset(val_bin, seq_len=seq_len)

    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = configure_optimizers(
        model,
        learning_rate=learning_rate,
        weight_decay=0.05,
        device_type=device.type,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    model.train()
    data_iter = iter(train_loader)
    best_val_loss = float("inf")
    warmup_steps = 10

    print("\n" + "-" * 72)
    print(f"{'Step':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val PPL':<9} | {'LR':<10} | {'VRAM (MB)':<8}")
    print("-" * 72, flush=True)

    out_file = PROJECT_ROOT / output_checkpoint
    out_file.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, steps + 1):
        lr = get_lr_cosine_warmup(
            step=step,
            warmup_steps=warmup_steps,
            max_steps=steps,
            peak_lr=learning_rate,
            min_lr=learning_rate * 0.1,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(x, targets=y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if step % 5 == 0 or step == steps:
            val_loss, val_ppl = evaluate_model(
                model,
                val_loader,
                eval_iters=8,
                device=str(device),
                mixed_precision=use_amp,
            )
            vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2) if device.type == "cuda" else 0.0
            print(
                f"{step:<8} | {loss.item():<12.4f} | {val_loss:<10.4f} | {val_ppl:<9.2f} | {lr:<10.2e} | {vram_mb:.1f} MB",
                flush=True,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "step": step,
                        "val_loss": val_loss,
                        "model_state_dict": model.state_dict(),
                        "config": {"model": config.model.to_dict()},
                    },
                    str(out_file),
                )
            model.train()

    print("-" * 72)
    print(f"Training Complete! Saved your checkpoint to: {output_checkpoint}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    return str(out_file)


def activate_in_server(checkpoint_path: str = "checkpoints/shree_chat_best.pt"):
    """Registers and activates your new checkpoint in the running server."""
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/providers/select",
            data=json.dumps({"provider": "local", "model": checkpoint_path}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[Server] Activated model: {data.get('active', {})}")
    except Exception as e:
        print(f"[Server] Note: Could not auto-select in server (it may be restarting): {e}")


def main():
    parser = argparse.ArgumentParser(description="Distill and train URA-Shree chat model.")
    parser.add_argument("--steps", type=int, default=60, help="Number of training steps")
    parser.add_argument("--count", type=int, default=25, help="Number of synthetic dialogues")
    args = parser.parse_args()

    # Step 1: Generate synthetic dialogues
    docs = generate_distillation_corpus(target_count=args.count)

    # Step 2: Prepare tokenized datasets
    train_bin, val_bin = prepare_datasets(docs)

    # Step 3: Train Shree model on GPU
    ckpt_path = train_shree_model(
        steps=args.steps,
        train_bin=train_bin,
        val_bin=val_bin,
    )

    # Step 4: Activate in Shree server
    activate_in_server("checkpoints/shree_chat_best.pt")
    print("\nAll done! Your own model 'shree_chat_best.pt' is trained and ready.")


if __name__ == "__main__":
    main()
