"""
Speculative decoding for URA-Shree.

A small draft model proposes K tokens autoregressively; the large target model
then checks all K in a single forward pass (`ShreeTransformerLM.step_multi`)
instead of K sequential ones. Tokens the target agrees with are kept for free;
the first disagreement is resampled from the target's own distribution, so the
output is drawn from exactly the same distribution the target would have
produced alone (Leviathan et al., 2023 / Chen et al., 2023) - speculation only
changes how many target forward passes that costs, never what comes out.

Greedy decoding (temperature <= 0) is handled as an exact special case: a
draft token is kept iff it equals the target's own argmax, which reproduces
the target's greedy output token-for-token.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from model.model import ShreeTransformerLM, PastKeyValues


def _truncate_cache(cache: PastKeyValues, length: int) -> PastKeyValues:
    return [(k[:, :, :length], v[:, :, :length]) for k, v in cache]


def _softmax_temp(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softmax(logits.float() / max(temperature, 1e-5), dim=-1)


@dataclass
class SpeculativeStats:
    """Running totals for the telemetry panel / tests."""

    proposed: int = 0
    accepted: int = 0
    rounds: int = 0

    @property
    def acceptance_rate(self) -> float:
        return round(self.accepted / self.proposed, 4) if self.proposed else 0.0

    def to_dict(self) -> dict:
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "rounds": self.rounds,
            "acceptance_rate": self.acceptance_rate,
        }


@torch.no_grad()
def speculative_generate(
    target: ShreeTransformerLM,
    draft: ShreeTransformerLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    num_speculative_tokens: int = 4,
    temperature: float = 0.0,
    eos_id: Optional[int] = None,
) -> Tuple[torch.Tensor, SpeculativeStats]:
    """
    Args:
        target: the large model whose output distribution we want.
        draft: a small, fast model used only to propose candidates.
        input_ids: [1, T] prompt tokens (batch size 1).
        num_speculative_tokens: draft tokens proposed per round (K).
        temperature: 0.0 selects the exact greedy special case.

    Returns:
        (generated token ids including the prompt [1, T + N], stats)
    """
    assert input_ids.size(0) == 1, "speculative_generate only supports batch size 1"
    device = input_ids.device
    greedy = temperature <= 0.0
    stats = SpeculativeStats()

    tokens = input_ids
    target_logits, target_cache = target.step(tokens)
    draft_logits, draft_cache = draft.step(tokens)

    while tokens.size(1) - input_ids.size(1) < max_new_tokens:
        k = min(num_speculative_tokens, max_new_tokens - (tokens.size(1) - input_ids.size(1)))
        prefix_len = target_cache[0][0].size(2)

        # -- draft proposes up to k tokens -----------------------------------
        draft_tokens: List[int] = []
        draft_dists: List[torch.Tensor] = []
        cur_logits, cur_cache = draft_logits, draft_cache
        for _ in range(k):
            if greedy:
                token_id = int(torch.argmax(cur_logits, dim=-1).item())
                probs = None
            else:
                probs = _softmax_temp(cur_logits, temperature)
                token_id = int(torch.multinomial(probs[0], 1).item())
            draft_tokens.append(token_id)
            draft_dists.append(probs[0] if probs is not None else None)
            step_in = torch.tensor([[token_id]], dtype=torch.long, device=device)
            cur_logits, cur_cache = draft.step(step_in, cur_cache)

        # -- target verifies all k in a single forward pass -------------------
        draft_block = torch.tensor([draft_tokens], dtype=torch.long, device=device)
        target_logits_all, target_cache_candidate = target.step_multi(draft_block, target_cache)

        accepted = 0
        extra_token: Optional[int] = None
        for i in range(k):
            verify_logits = target_logits if i == 0 else target_logits_all[:, i - 1, :]
            x_i = draft_tokens[i]

            if greedy:
                target_argmax = int(torch.argmax(verify_logits, dim=-1).item())
                if target_argmax == x_i:
                    accepted += 1
                    continue
                extra_token = target_argmax
                break

            p_i = _softmax_temp(verify_logits, temperature)[0]
            q_i = draft_dists[i]
            accept_prob = min(1.0, (p_i[x_i] / q_i[x_i].clamp_min(1e-8)).item())
            if random.random() < accept_prob:
                accepted += 1
                continue
            residual = (p_i - q_i).clamp_min(0)
            total = residual.sum()
            residual = residual / total if total > 0 else p_i
            extra_token = int(torch.multinomial(residual, 1).item())
            break

        stats.proposed += k
        stats.accepted += accepted
        stats.rounds += 1

        if extra_token is None:
            # Every draft token was accepted; the bonus token is free - the
            # target already computed its distribution one position past the
            # last accepted draft token.
            if greedy:
                extra_token = int(torch.argmax(target_logits_all[:, k - 1, :], dim=-1).item())
            else:
                extra_token = int(
                    torch.multinomial(_softmax_temp(target_logits_all[:, k - 1, :], temperature)[0], 1).item()
                )

        new_tokens = draft_tokens[:accepted] + [extra_token]

        # `k` was already clamped to the remaining budget, but the bonus/
        # correction token is one more than that - drop it if it would
        # overshoot max_new_tokens.
        remaining = max_new_tokens - (tokens.size(1) - input_ids.size(1))
        budget_exhausted = len(new_tokens) > remaining
        if budget_exhausted:
            new_tokens = new_tokens[:remaining]

        if eos_id is not None and eos_id in new_tokens:
            cut = new_tokens.index(eos_id) + 1
            new_tokens = new_tokens[:cut]
            tokens = torch.cat([tokens, torch.tensor([new_tokens], dtype=torch.long, device=device)], dim=1)
            break

        tokens = torch.cat([tokens, torch.tensor([new_tokens], dtype=torch.long, device=device)], dim=1)

        if budget_exhausted:
            break

        keep = prefix_len + accepted
        extra_in = torch.tensor([[extra_token]], dtype=torch.long, device=device)
        target_logits, target_cache = target.step(extra_in, _truncate_cache(target_cache_candidate, keep))
        draft_logits, draft_cache = draft.step(extra_in, _truncate_cache(cur_cache, keep))

    return tokens, stats
