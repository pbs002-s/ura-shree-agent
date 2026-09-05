# Training

Two paths lead to a checkpoint. The first trains the model from nothing on
this repository's own source. The second distils a local teacher into it so it
answers questions instead of predicting Python.

## From scratch

```powershell
python -m tokenizer.train_tokenizer --vocab-size 4096   # learns merges from the source tree
python -m data.prepare_dataset                          # builds train.bin and val.bin
python -m training.train --config configs/small.yaml
python -m training.train_coding --steps 200             # supervised fine-tune
python -m inference.chat --model checkpoints/coding_best.pt
```

`configs/` holds `small` (11M), `medium` (82M) and `large` (283M). Training
writes `checkpoints/best.pt` on the best validation loss and `last.pt` every
`save_interval` steps.

The honest limit is data. The corpus is this repository plus a seed set -
about 320,000 tokens. An 11M-parameter model wants a few hundred million.
Point `data/source_corpus.py` at a larger tree to change that.

## Distillation

A model trained on source code continues source code. Asked "what is
recursion", it emits a plausible function definition, because that is what
follows a line of text in its training data. The fix is not more parameters,
it is examples of the shape you actually want.

`scripts/distill_and_train.py` produces them locally:

```powershell
ollama pull qwen3.5:0.8b
python scripts/distill_and_train.py --count 25 --steps 60
```

Four stages:

1. **Generate.** Prompts a local Ollama teacher across a seed set spanning
   general explanation, Python, JavaScript, algorithms and debugging. Replies
   are cached to `datasets/distilled_chat.json` and only missing prompts are
   re-run, so an interrupted pass resumes instead of restarting.
2. **Format and tokenise.** Each exchange becomes one boundary-tagged
   document -

   ```
   <|bos|><|system|>…<|user|>…<|assistant|>…<|eos|>
   ```

   - encoded with the project's own BPE and written as `uint16` to
   `datasets/shree_chat_train.bin` and `shree_chat_val.bin`. The documents are
   replicated to give the run enough token density to converge on a corpus
   this small.
3. **Train.** Loads `checkpoints/best.pt` as the starting point and fine-tunes
   with BF16 autocast, cosine decay and warmup, writing
   `checkpoints/shree_chat_best.pt`.
4. **Activate.** Calls `/api/providers/select` on a server running at
   `127.0.0.1:8000` to make the new checkpoint the active model. If nothing is
   listening it says so and carries on - the checkpoint is already on disk and
   selectable from Settings.

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--count` | 25 | Seed prompts to distil |
| `--steps` | 60 | Fine-tuning steps |

No hosted API is called and no third-party weights are shipped - only the
teacher's own text, which is what keeps the resulting checkpoint
distributable. The cached corpus is committed so a training run reproduces
without a teacher installed.

The scale is honest about itself: 25 dialogues will not make an 11M-parameter
model useful. It makes it answer in the right *shape*, which is the part the
architecture can learn and the corpus size cannot fake. Raise `--count` and
extend `SEED_PROMPTS` to go further.

## The Ollama persona

`Modelfile` is the other half of the same idea, applied to a model that is
already capable:

```powershell
ollama create shree -f Modelfile
```

It sets the system prompt - the same identity rule the agent uses, including
the instruction not to mention its provenance unless asked - and carries
few-shot messages demonstrating a direct coding answer with no preamble. The
result registers as `shree:latest` and is selectable through the Ollama
provider.

Few-shot messages rather than instructions alone, because "do not introduce
yourself" is a rule a small model breaks and a demonstrated pattern is one it
imitates.

## Custom Datasets & System Prompt Leaks Training

For comprehensive domain conditioning and identity alignment, `scripts/train_custom_datasets.py` trains the model on curated agent datasets:

```powershell
python scripts/train_custom_datasets.py --config configs/medium.yaml --steps 300 --max-did 25000
```

### Ingested Data Streams:
1. **`dataset/greetings_identity.jsonl`**: 148 prompt-response pairs teaching Shree its core identity, creator attribution (Pritam from DIU), PACES architecture, and handling typos (`hwllo`, `what is tour name`, `ho made you`, `what is paces`).
2. **`dataset/clean_coding_dialogues.jsonl`**: Technical assistant dialogues covering Python, Git, JavaScript, debugging, and software engineering.
3. **`system_prompts_leaks-main`**: 402 real-world system prompt and instruction sets from Anthropic, Google, Cursor, OpenAI, DeepSeek, and Qwen, chunked into ~2,000-character segments.
4. **`dataset/DID_500K_dataset.json`**: 25,000 clean Decentralized Identity verification documents.
5. **`data/coding_corpus.py` & `data/prepare_dataset.py`**: Agent tool traces and foundational coding algorithms.

This pipeline enforces mixed-precision (BF16), linear warmup with cosine annealing, automatic checkpoint preservation (`checkpoints/medium_best.pt` & `checkpoints/best.pt`), and live generation benchmarking.
