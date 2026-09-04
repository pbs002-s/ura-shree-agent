"""
Terminal chat client for the local URA-Shree model.

The generation machinery lives in `inference.engine`; this module is the
command-line shell over it and re-exports `InferenceEngine` so existing
imports of `inference.chat` keep working.
"""

import os
import sys
import argparse
from pathlib import Path

# Re-launch under the project virtualenv when torch is missing from the
# interpreter that was actually invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    try:
        import torch  # noqa: F401
    except ImportError:
        import subprocess

        sys.exit(subprocess.call([str(_venv_python)] + sys.argv))

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

from inference.engine import InferenceEngine, GenerationStats  # noqa: E402
from agent.prompts import LOCAL_MODEL_SYSTEM_PROMPT  # noqa: E402

__all__ = ["InferenceEngine", "GenerationStats", "start_interactive_chat", "main"]


def start_interactive_chat(
    engine: InferenceEngine,
    temperature: float = 0.7,
    max_tokens: int = 400,
) -> None:
    """Runs a REPL against the local checkpoint."""
    info = engine.describe()
    print()
    print("=" * 68)
    print("  URA-Shree - local model chat")
    print(f"  Device {engine.device} | {info['parameters']:,} parameters "
          f"| {info['memory']['total_mb']} MB resident")
    print(f"  Checkpoint {engine.checkpoint_path} | step {engine.step_count} "
          f"| val loss {engine.val_loss:.4f}")
    print("=" * 68)
    print("  /quit   end the session")
    print("  /clear  reset the conversation")
    print("  /temp <value>  change sampling temperature")
    print("  /stats  timing for the last reply")
    print("-" * 68)
    print()

    system = f"<|system|>\n{LOCAL_MODEL_SYSTEM_PROMPT}\n"
    history = system

    while True:
        try:
            user_input = input("you > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            return

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in ("/quit", "/exit"):
            print("Session ended.")
            return
        if lowered == "/clear":
            history = system
            print("[conversation cleared]\n")
            continue
        if lowered == "/stats":
            print(f"[{engine.last_stats.to_dict()}]\n")
            continue
        if lowered.startswith("/temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"[temperature = {temperature}]\n")
            except (IndexError, ValueError):
                print("[usage: /temp 0.7]\n")
            continue

        prompt = history + f"<|user|>\n{user_input}\n<|assistant|>\n"
        print("shree > ", end="", flush=True)

        reply = ""
        for piece in engine.generate_stream(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        ):
            print(piece, end="", flush=True)
            reply += piece

        stats = engine.last_stats
        print(f"\n\n[{stats.completion_tokens} tokens, {stats.tokens_per_second} tok/s]\n")
        history = prompt + reply + "\n<|eos|>\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the local URA-Shree model")
    parser.add_argument("--model", "--checkpoint", dest="model", type=str,
                        default="checkpoints/coding_best.pt", help="path to a .pt checkpoint")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    parser.add_argument("--prompt", type=str, default=None, help="one-shot prompt, then exit")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / mps")
    parser.add_argument("--quantize", action="store_true",
                        help="int8-quantise the linear layers (CPU only, ~4x less RAM)")
    parser.add_argument("--compile", dest="compile_model", action="store_true",
                        help="run torch.compile on the model (slow first call, faster after)")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        for fallback in ("checkpoints/best.pt", "checkpoints/last.pt"):
            if os.path.exists(fallback):
                model_path = fallback
                break

    engine = InferenceEngine(
        checkpoint_path=model_path,
        tokenizer_path=args.tokenizer,
        device=args.device,
        quantize=args.quantize,
        compile_model=args.compile_model,
    )

    if args.prompt:
        prompt = f"<|system|>\n{LOCAL_MODEL_SYSTEM_PROMPT}\n<|user|>\n{args.prompt}\n<|assistant|>\n"
        for piece in engine.generate_stream(
            prompt=prompt, max_new_tokens=args.max_tokens, temperature=args.temperature
        ):
            print(piece, end="", flush=True)
        print()
    else:
        start_interactive_chat(engine, temperature=args.temperature, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
