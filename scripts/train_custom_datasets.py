"""
Training & Evaluation Pipeline for Custom Uploaded Datasets.
Ingests:
  1. dataset/train.jsonl (conversational & assistant fine-tuning)
  2. dataset/DID_500K_dataset.json (decentralized identity records)
  3. dataset/greetings_identity.jsonl (greetings, identity, creator, PACES, capabilities, typos)

Tokenizes into memory-mapped uint16 binary format, evaluates pre-training baseline,
trains on NVIDIA RTX 4060 GPU with BF16, and computes improvement metrics.
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import torch

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ProjectConfig, ModelConfig
from model.model import ShreeTransformerLM
from tokenizer.tokenizer import BPETokenizer
from data.dataset import CausalLMDataset, create_dataloader
from training.evaluate import evaluate_model
from training.train import configure_optimizers, get_lr_cosine_warmup
from data.coding_corpus import CODING_TRACES
from data.prepare_dataset import SEED_DOCUMENTS


BENCHMARK_PROMPTS = [
    ("hi", "<|bos|><|user|>\nhi\n<|assistant|>\n"),
    ("hwllo", "<|bos|><|user|>\nhwllo\n<|assistant|>\n"),
    ("what is tour name", "<|bos|><|user|>\nwhat is tour name\n<|assistant|>\n"),
    ("ho made you", "<|bos|><|user|>\nho made you\n<|assistant|>\n"),
    ("what is paces", "<|bos|><|user|>\nwhat is paces\n<|assistant|>\n"),
    ("what dose you do", "<|bos|><|user|>\nwhat dose you do\n<|assistant|>\n"),
    ("can you help me code", "<|bos|><|user|>\ncan you help me code\n<|assistant|>\n"),
    ("bye", "<|bos|><|user|>\nbye\n<|assistant|>\n"),
]


def prepare_data(
    train_jsonl_path: str = "dataset/train.jsonl",
    did_json_path: str = "dataset/DID_500K_dataset.json",
    greetings_jsonl_path: str = "dataset/greetings_identity.jsonl",
    tokenizer_path: str = "checkpoints/tokenizer.json",
    output_dir: str = "datasets",
    max_did_samples: int = 100000,
    val_ratio: float = 0.05,
    force_prep: bool = False,
) -> Tuple[str, str, int, int]:
    """
    Parses and tokenizes all datasets into memory-mapped uint16 binary arrays.
    """
    out_path = PROJECT_ROOT / output_dir
    train_bin = out_path / "train.bin"
    val_bin = out_path / "val.bin"

    if not force_prep and train_bin.exists() and val_bin.exists() and train_bin.stat().st_size > 1000000:
        n_train = train_bin.stat().st_size // 2
        n_val = val_bin.stat().st_size // 2
        print(f"[Prep] Found existing tokenized datasets:")
        print(f"  - train.bin: {train_bin.stat().st_size / (1024**2):.2f} MB ({n_train:,} tokens)")
        print(f"  - val.bin  : {val_bin.stat().st_size / (1024**2):.2f} MB ({n_val:,} tokens)")
        return str(train_bin), str(val_bin), n_train, n_val

    tok_file = PROJECT_ROOT / tokenizer_path
    if not tok_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_file}")
    tokenizer = BPETokenizer.load(str(tok_file))
    print(f"[Prep] Loaded tokenizer with {len(tokenizer)} tokens.")

    documents: List[str] = []

    # 1. Process dataset/greetings_identity.jsonl (high-priority identity & greetings)
    greet_file = PROJECT_ROOT / greetings_jsonl_path
    if greet_file.exists():
        print(f"[Prep] Loading greetings & identity dataset from {greetings_jsonl_path}...")
        t0 = time.time()
        greet_count = 0
        greet_docs: List[str] = []
        with open(greet_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    user_msg = ""
                    asst_msg = ""
                    for m in data.get("messages", []):
                        if m.get("role") == "user":
                            user_msg = m.get("content", "").strip()
                        elif m.get("role") == "assistant":
                            asst_msg = m.get("content", "").strip()
                    if user_msg and asst_msg:
                        # Variant A: Standard system prompt
                        doc_a = (
                            "<|bos|><|system|>\n"
                            "You are Shree, an intelligent and helpful AI assistant.\n"
                            f"<|user|>\n{user_msg}\n"
                            f"<|assistant|>\n{asst_msg}\n<|eos|>"
                        )
                        # Variant B: Direct user turn (common in chat interfaces)
                        doc_b = (
                            "<|bos|><|user|>\n"
                            f"{user_msg}\n"
                            f"<|assistant|>\n{asst_msg}\n<|eos|>"
                        )
                        # Variant C: Local agent compact prompt
                        doc_c = (
                            "<|bos|><|system|>\n"
                            "You are Shree, a local coding assistant. Answer briefly and directly. "
                            "Use code blocks for code. No preamble, no filler, no emoji.\n"
                            f"<|user|>\n{user_msg}\n"
                            f"<|assistant|>\n{asst_msg}\n<|eos|>"
                        )
                        greet_docs.extend([doc_a, doc_b, doc_c])
                        greet_count += 1
                except Exception:
                    continue
        
        # Replicate greetings 35x so every gradient step across the run encounters these patterns
        upsampled_greets = greet_docs * 35
        documents.extend(upsampled_greets)
        print(f"[Prep] Loaded {greet_count} unique greeting/identity entries -> {len(upsampled_greets):,} upsampled documents ({time.time() - t0:.2f}s).")
    else:
        print(f"[Prep] Warning: {greetings_jsonl_path} not found. Continuing without it.")

    # 2. Process clean technical & coding assistant dialogues (NO REDDIT JOKES)
    clean_dialogue_file = PROJECT_ROOT / "dataset" / "clean_coding_dialogues.jsonl"
    if clean_dialogue_file.exists():
        print("[Prep] Loading clean technical dialogues from dataset/clean_coding_dialogues.jsonl...")
        t0 = time.time()
        clean_docs: List[str] = []
        with open(clean_dialogue_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    user_msg = ""
                    asst_msg = ""
                    for m in data.get("messages", []):
                        if m.get("role") == "user":
                            user_msg = m.get("content", "").strip()
                        elif m.get("role") == "assistant":
                            asst_msg = m.get("content", "").strip()
                    if user_msg and asst_msg:
                        # Variant A: Standard assistant
                        doc_a = (
                            "<|bos|><|system|>\n"
                            "You are Shree, an intelligent and helpful AI assistant.\n"
                            f"<|user|>\n{user_msg}\n"
                            f"<|assistant|>\n{asst_msg}\n<|eos|>"
                        )
                        # Variant B: Direct user turn
                        doc_b = (
                            "<|bos|><|user|>\n"
                            f"{user_msg}\n"
                            f"<|assistant|>\n{asst_msg}\n<|eos|>"
                        )
                        # Variant C: Local agent compact
                        doc_c = (
                            "<|bos|><|system|>\n"
                            "You are Shree, a local coding assistant. Answer briefly and directly. "
                            "Use code blocks for code. No preamble, no filler, no emoji.\n"
                            f"<|user|>\n{user_msg}\n"
                            f"<|assistant|>\n{asst_msg}\n<|eos|>"
                        )
                        clean_docs.extend([doc_a, doc_b, doc_c])
                except Exception:
                    continue
        upsampled_clean = clean_docs * 40
        documents.extend(upsampled_clean)
        print(f"[Prep] Loaded clean technical dialogues -> {len(upsampled_clean):,} upsampled documents ({time.time() - t0:.2f}s).")

    # Coding agent traces & seed algorithms
    for trace in CODING_TRACES:
        documents.extend([trace] * 400)
    for seed in SEED_DOCUMENTS:
        doc = f"<|bos|>{seed.strip()}<|eos|>"
        documents.extend([doc] * 120)
    print(f"[Prep] Loaded coding agent traces and seed algorithms.")

    # 3. Process dataset/DID_500K_dataset.json (Decentralized Identity records)
    did_file = PROJECT_ROOT / did_json_path
    if did_file.exists():
        print(f"[Prep] Loading Decentralized Identity dataset from {did_json_path} (limit={max_did_samples:,})...")
        t0 = time.time()
        did_count = 0
        with open(did_file, "r", encoding="utf-8") as f:
            did_data = json.load(f)
            sample_slice = did_data[:max_did_samples] if max_did_samples > 0 else did_data
            for item in sample_slice:
                did_id = item.get("id", "")
                formatted_item = json.dumps(item, ensure_ascii=False)

                doc = (
                    "<|bos|><|system|>\n"
                    "You are Shree, an autonomous AI assistant specialized in Decentralized Identity (DID) security.\n"
                    f"<|user|>\nInspect and verify DID document: {did_id}\n"
                    f"<|assistant|>\n{formatted_item}\n<|eos|>"
                )
                documents.append(doc)
                did_count += 1
        print(f"[Prep] Loaded {did_count:,} DID identity documents ({time.time() - t0:.2f}s).")
    else:
        print(f"[Prep] Warning: {did_json_path} not found.")

    print(f"[Prep] Total combined corpus documents: {len(documents):,}. Shuffling and tokenizing...")
    t0 = time.time()

    # Shuffle documents to interleave greetings, chat, and DID records
    np.random.seed(42)
    np.random.shuffle(documents)

    corpus_tokens: List[int] = []
    for i, doc in enumerate(documents):
        corpus_tokens.extend(tokenizer.encode(doc))
        if (i + 1) % 50000 == 0:
            print(f"  Tokenized {i + 1:,} / {len(documents):,} documents ({len(corpus_tokens):,} tokens)...")

    total_tokens = len(corpus_tokens)
    elapsed = time.time() - t0
    tok_speed = total_tokens / max(elapsed, 0.001)
    print(f"[Prep] Tokenization complete! Total tokens: {total_tokens:,} ({elapsed:.1f}s, {tok_speed:,.0f} tok/s).")

    # Split train and validation
    split_idx = int(total_tokens * (1.0 - val_ratio))
    train_tokens = corpus_tokens[:split_idx]
    val_tokens = corpus_tokens[split_idx:]

    out_path.mkdir(parents=True, exist_ok=True)

    np.array(train_tokens, dtype=np.uint16).tofile(str(train_bin))
    np.array(val_tokens, dtype=np.uint16).tofile(str(val_bin))

    print(f"[Prep] Saved train.bin: {train_bin.stat().st_size / (1024**2):.2f} MB ({len(train_tokens):,} tokens)")
    print(f"[Prep] Saved val.bin  : {val_bin.stat().st_size / (1024**2):.2f} MB ({len(val_tokens):,} tokens)")

    return str(train_bin), str(val_bin), len(train_tokens), len(val_tokens)


def sample_generation(
    model: ShreeTransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
    temperature: float = 0.6,
) -> str:
    """Generates sample response string for quality verification."""
    model.eval()
    try:
        input_ids = tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        eos_id = tokenizer.vocab.token_to_id.get("<|eos|>", 2)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if x.size(1) >= model.config.max_seq_len:
                    break
                logits, _ = model(x[:, -model.config.max_seq_len:])
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, next_token), dim=1)
                if next_token.item() == eos_id:
                    break

        output_tokens = x[0].tolist()
        gen_tokens = output_tokens[len(input_ids):]
        res = tokenizer.decode(gen_tokens).strip()
        # Clean up any trailing eos marker
        for stop_token in ["<|eos|>", "<|user|>", "<|system|>"]:
            if stop_token in res:
                res = res.split(stop_token)[0].strip()
        return res
    except Exception as e:
        return f"[Generation error: {e}]"


def run_training_pipeline(
    steps: int = 200,
    max_did: int = 100000,
    base_checkpoint: str = "checkpoints/best.pt",
    output_checkpoint: str = "checkpoints/custom_best.pt",
    learning_rate: float = 3.0e-4,
    batch_size: int = 16,
    grad_accum_steps: int = 4,
    force_prep: bool = True,
):
    print("=" * 80)
    print("      URA-Shree: Training on Custom Greetings & Identity Datasets")
    print("=" * 80)

    # Step 1: Prepare data
    train_bin, val_bin, n_train, n_val = prepare_data(
        max_did_samples=max_did,
        val_ratio=0.05,
        force_prep=force_prep,
    )

    # Step 2: Initialize hardware and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Hardware] Compute device: {device}")
    if device.type == "cuda":
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = ProjectConfig.load_from_yaml(str(PROJECT_ROOT / "configs/small.yaml"))
    tokenizer = BPETokenizer.load(str(PROJECT_ROOT / "checkpoints/tokenizer.json"))

    model = ShreeTransformerLM(config.model).to(device)

    # Load baseline weights
    base_path = PROJECT_ROOT / base_checkpoint
    if base_path.exists():
        print(f"[Model] Loading baseline checkpoint from: {base_checkpoint}")
        ckpt = torch.load(str(base_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        print("[Model] Starting training from scratch (no base checkpoint found).")

    # Data loaders
    seq_len = config.model.max_seq_len
    train_ds = CausalLMDataset(train_bin, seq_len=seq_len)
    val_ds = CausalLMDataset(val_bin, seq_len=seq_len)

    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    use_amp = device.type == "cuda"

    # Baseline Evaluation before training
    print("\n[Baseline] Evaluating model on new dataset BEFORE training...")
    initial_val_loss, initial_val_ppl = evaluate_model(
        model, val_loader, eval_iters=25, device=str(device), mixed_precision=use_amp
    )
    print(f"[Baseline] Initial Val Loss: {initial_val_loss:.4f} | Initial Val Perplexity: {initial_val_ppl:.2f}")

    # Benchmark generations BEFORE training
    print("\n" + "=" * 80)
    print("       MODEL GENERATION BENCHMARK (BEFORE TRAINING)")
    print("=" * 80)
    baseline_samples: Dict[str, str] = {}
    for prompt_label, prompt_text in BENCHMARK_PROMPTS:
        resp = sample_generation(model, tokenizer, prompt_text, device, max_new_tokens=45)
        baseline_samples[prompt_label] = resp
        print(f"User: {prompt_label:<20} -> Shree: {resp}")

    # Step 3: Optimizer & Scheduler Setup
    optimizer = configure_optimizers(
        model,
        learning_rate=learning_rate,
        weight_decay=0.05,
        device_type=device.type,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    warmup_steps = 20
    eval_interval = 25

    print("\n" + "-" * 90)
    print(f"{'Step':<8} | {'Train Loss':<11} | {'Val Loss':<10} | {'Val PPL':<9} | {'LR':<10} | {'Tok/Sec':<9} | {'VRAM':<8}")
    print("-" * 90, flush=True)

    model.train()
    data_iter = iter(train_loader)
    best_val_loss = initial_val_loss
    history = []

    t_start = time.time()
    step = 0
    tokens_per_step = batch_size * grad_accum_steps * seq_len

    while step < steps:
        step += 1
        t_step = time.time()

        # Update LR with cosine warmup
        lr = get_lr_cosine_warmup(
            step=step,
            warmup_steps=warmup_steps,
            max_steps=steps,
            peak_lr=learning_rate,
            min_lr=learning_rate * 0.1,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        micro_loss_sum = 0.0

        for _ in range(grad_accum_steps):
            try:
                x_b, y_b = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x_b, y_b = next(data_iter)

            x_b = x_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x_b, targets=y_b)
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                _, loss = model(x_b, targets=y_b)
                loss = loss / grad_accum_steps
                loss.backward()

            micro_loss_sum += loss.item() * grad_accum_steps

        # Clip gradients
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        elapsed_step = time.time() - t_step
        tok_sec = tokens_per_step / max(elapsed_step, 0.001)
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

        # Periodic Evaluation
        if step % eval_interval == 0 or step == steps:
            val_loss, val_ppl = evaluate_model(
                model, val_loader, eval_iters=20, device=str(device), mixed_precision=use_amp
            )
            model.train()

            print(f"{step:<8} | {micro_loss_sum:<11.4f} | {val_loss:<10.4f} | {val_ppl:<9.2f} | {lr:<10.2e} | {tok_sec:<9.0f} | {vram_mb:<7.0f}MB", flush=True)

            history.append({
                "step": step,
                "train_loss": micro_loss_sum,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            })

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_payload = {
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "model": config.model.__dict__,
                        **config.model.__dict__,
                    },
                    "timestamp": time.time(),
                }
                torch.save(save_payload, str(PROJECT_ROOT / output_checkpoint))
                torch.save(save_payload, str(PROJECT_ROOT / "checkpoints/best.pt"))
        else:
            if step % 5 == 0:
                print(f"{step:<8} | {micro_loss_sum:<11.4f} | {'--':<10} | {'--':<9} | {lr:<10.2e} | {tok_sec:<9.0f} | {vram_mb:<7.0f}MB", flush=True)

    total_training_time = time.time() - t_start

    # Final Evaluation
    final_val_loss, final_val_ppl = evaluate_model(
        model, val_loader, eval_iters=30, device=str(device), mixed_precision=use_amp
    )

    # Always ensure the final weights are saved
    final_payload = {
        "step": step,
        "val_loss": final_val_loss,
        "val_ppl": final_val_ppl,
        "model_state_dict": model.state_dict(),
        "config": {
            "model": config.model.__dict__,
            **config.model.__dict__,
        },
        "timestamp": time.time(),
    }
    torch.save(final_payload, str(PROJECT_ROOT / output_checkpoint))
    torch.save(final_payload, str(PROJECT_ROOT / "checkpoints/best.pt"))

    # Benchmark generations AFTER training
    print("\n" + "=" * 80)
    print("       MODEL GENERATION BENCHMARK (AFTER TRAINING)")
    print("=" * 80)
    after_samples: Dict[str, str] = {}
    for prompt_label, prompt_text in BENCHMARK_PROMPTS:
        resp = sample_generation(model, tokenizer, prompt_text, device, max_new_tokens=45)
        after_samples[prompt_label] = resp
        print(f"User: {prompt_label:<20} -> Shree: {resp}")

    # Summary of Improvements
    loss_improvement = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100
    ppl_improvement = ((initial_val_ppl - final_val_ppl) / initial_val_ppl) * 100

    report = {
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "loss_improvement_pct": loss_improvement,
        "initial_val_ppl": initial_val_ppl,
        "final_val_ppl": final_val_ppl,
        "ppl_improvement_pct": ppl_improvement,
        "total_training_time_s": total_training_time,
        "total_steps": steps,
        "best_val_loss": best_val_loss,
        "checkpoint_saved": output_checkpoint,
        "benchmark_comparison": [
            {
                "prompt": label,
                "before": baseline_samples.get(label, ""),
                "after": after_samples.get(label, ""),
            }
            for label, _ in BENCHMARK_PROMPTS
        ]
    }

    # Save training report JSON
    with open(PROJECT_ROOT / "checkpoints/training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("                    TRAINING IMPROVEMENT REPORT")
    print("=" * 80)
    print(f"Validation Loss       : {initial_val_loss:.4f}  -->  {final_val_loss:.4f}  ({loss_improvement:+.2f}% improvement)")
    print(f"Validation Perplexity : {initial_val_ppl:.2f}    -->  {final_val_ppl:.2f}    ({ppl_improvement:+.2f}% improvement)")
    print(f"Total Steps Completed : {steps}")
    print(f"Total Training Time   : {total_training_time / 60:.2f} minutes")
    print(f"Best Model Checkpoint : {output_checkpoint} & checkpoints/best.pt")
    print("=" * 80)

    # Print comparison table
    print("\n[PROMPT RESPONSE COMPARISON]:")
    for item in report["benchmark_comparison"]:
        print(f"\nQuery: '{item['prompt']}'")
        print(f"  BEFORE: {item['before']}")
        print(f"  AFTER : {item['after']}")

    # Activate in server if server is running
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/providers/select",
            data=json.dumps({"provider": "local", "model": output_checkpoint}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"\n[Server] Activated model in running Shree server: {data.get('active', {})}")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on uploaded custom datasets.")
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument("--max-did", type=int, default=100000, help="Max DID records to include")
    parser.add_argument("--lr", type=float, default=3.0e-4, help="Peak learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro batch size")
    parser.add_argument("--force-prep", action="store_true", default=False, help="Force rebuild binary datasets")
    args = parser.parse_args()

    run_training_pipeline(
        steps=args.steps,
        max_did=args.max_did,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        force_prep=args.force_prep,
    )
