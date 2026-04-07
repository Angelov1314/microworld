"""
Phase 4: Random Shooting Planner (Model-Predictive Control)

At each step:
    1. Sample N random action sequences of length H
    2. Roll each sequence out through the GRU world model (frozen)
    3. Score by cumulative discounted reward
    4. Execute the first action of the best sequence in the TRUE engine

Reward function (from 甄嬛's perspective):
    mood_score(甄嬛)        — higher mood = higher reward
    energy(甄嬛)            — maintain energy
    rel(甄嬛→皇上)          — build imperial favor
"""

from __future__ import annotations
import random
import torch
import numpy as np
from typing import Optional

from core.world import (
    Action, LOCATIONS, MOODS, GOALS, FACTS, ACTION_TYPES, CHARACTERS_FULL,
)
from core.tensorize import CHAR_DIM, rel_dim, encode_action
from models.baseline_mlp import ActionEncoder


# ─── Reward ───────────────────────────────────────────────────────────────────

# Mood score: how positive each mood is for 甄嬛
# MOODS = ["calm", "anxious", "angry", "happy", "sad"]
MOOD_SCORES = torch.tensor([0.6, 0.2, 0.0, 1.0, 0.1], dtype=torch.float32)

# Relationship index for 甄嬛→皇上 in the flat rel vector
# chars = [甄嬛(0), 皇后(1), 安陵容(2), 皇上(3), 沈眉庄(4)]
# pairs = [(a,b) for a in chars for b in chars if a!=b]  sorted by a then b
# (0,1)=0, (0,2)=1, (0,3)=2, (0,4)=3  → 甄嬛→皇上 = index 2
_ZH_EMPEROR_REL_IDX = 2


def compute_reward(pred_flat: torch.Tensor, n_chars: int) -> torch.Tensor:
    """
    Compute scalar reward from decoded flat state (B,) tensor.

    Args:
        pred_flat: (B, flat_state_dim)  VAE-decoded prediction
        n_chars:   number of characters

    Returns:
        reward: (B,)  scalar per sample
    """
    n_rels   = rel_dim(n_chars)
    char_end = n_chars * CHAR_DIM
    rel_end  = char_end + n_rels

    # 甄嬛 is character index 0
    pred_chars = pred_flat[:, :char_end].view(-1, n_chars, CHAR_DIM)
    pred_rels  = pred_flat[:, char_end:rel_end]

    # Mood score (apply softmax to get distribution, dot with MOOD_SCORES)
    loc_end  = len(LOCATIONS)
    mood_end = loc_end + len(MOODS)
    mood_logits = pred_chars[:, 0, loc_end:mood_end]       # (B, 5)
    mood_probs  = torch.softmax(mood_logits, dim=-1)        # (B, 5)
    mood_score  = (mood_probs * MOOD_SCORES.to(mood_probs.device)).sum(dim=-1)  # (B,)

    # Energy (clamp to [0,1])
    energy = pred_chars[:, 0, mood_end].clamp(0.0, 1.0)    # (B,)

    # Relationship with 皇上 (normalize [-1,1] → [0,1])
    rel_emperor = (pred_rels[:, _ZH_EMPEROR_REL_IDX] + 1.0) / 2.0  # (B,)

    # Weighted reward
    reward = 0.5 * mood_score + 0.3 * energy + 0.2 * rel_emperor
    return reward  # (B,)


# ─── Action Sampling ──────────────────────────────────────────────────────────

def sample_random_action(actor: str, characters: list[str],
                          rng: random.Random) -> Action:
    """Sample a random valid-ish action for `actor`."""
    atype = rng.choice(ACTION_TYPES)
    others = [c for c in characters if c != actor]

    if atype == "move_to":
        return Action(type=atype, actor=actor, target_loc=rng.choice(LOCATIONS))
    elif atype in ("talk_to", "argue", "cooperate"):
        return Action(type=atype, actor=actor, target_char=rng.choice(others))
    elif atype == "inform":
        return Action(type=atype, actor=actor,
                      target_char=rng.choice(others),
                      target_fact=rng.choice(FACTS))
    elif atype == "plan":
        return Action(type=atype, actor=actor, target_goal=rng.choice(GOALS))
    else:  # rest, observe, wait, react
        return Action(type=atype, actor=actor)


def encode_action_batch(actions: list[Action], characters: list[str],
                         actor: str, device: str) -> dict:
    """
    Encode a list of N actions into batched tensors for the GRU model.
    Returns dict of (N,) tensors.
    """
    actor_idx = torch.tensor(characters.index(actor), dtype=torch.long)
    encs = [encode_action(a, characters) for a in actions]
    return {
        "action_type": torch.stack([e["type"]        for e in encs]).to(device),
        "action_char": torch.stack([e["target_char"] for e in encs]).to(device),
        "action_loc":  torch.stack([e["target_loc"]  for e in encs]).to(device),
        "action_fact": torch.stack([e["target_fact"] for e in encs]).to(device),
        "action_goal": torch.stack([e["target_goal"] for e in encs]).to(device),
        "actor_idx":   actor_idx.expand(len(actions)).to(device),
    }


# ─── Planner ──────────────────────────────────────────────────────────────────

class RandomShootingPlanner:
    """
    MPC via random shooting:
        For each planning step, sample N candidate sequences of length H,
        roll out in world model, pick the sequence with highest discounted reward.

    Args:
        gru_model:   GRUWorldModel (frozen)
        vae:         WorldStateVAE (frozen)
        n_chars:     number of characters
        n_samples:   candidate action sequences per step
        horizon:     rollout length (steps)
        gamma:       discount factor
        actor:       which character the planner controls (default: 甄嬛)
        characters:  ordered character list
        device:      torch device
    """

    def __init__(
        self,
        gru_model,
        vae,
        n_chars: int,
        n_samples: int = 200,
        horizon: int = 5,
        gamma: float = 0.95,
        actor: str = "甄嬛",
        characters: Optional[list[str]] = None,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.model      = gru_model
        self.vae        = vae
        self.n_chars    = n_chars
        self.n_samples  = n_samples
        self.horizon    = horizon
        self.gamma      = gamma
        self.actor      = actor
        self.characters = characters or CHARACTERS_FULL
        self.device     = device
        self.rng        = random.Random(seed)

        self.model.eval()
        self.vae.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.vae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def plan(self, z: torch.Tensor, h: Optional[torch.Tensor]) -> tuple[Action, torch.Tensor]:
        """
        Choose the best action for current latent state z.

        Args:
            z: (z_dim,)        current latent
            h: (1,1,gru_hidden) or None  GRU hidden state

        Returns:
            best_action: Action  to execute in true engine
            h_next:      (1,1,gru_hidden)  hidden state after first step
        """
        N   = self.n_samples
        dev = self.device

        # Sample N action sequences of length H
        seqs = [
            [sample_random_action(self.actor, self.characters, self.rng)
             for _ in range(self.horizon)]
            for _ in range(N)
        ]

        # Broadcast z and h to (N, z_dim) / (1, N, gru_hidden)
        z_batch = z.unsqueeze(0).expand(N, -1).clone()  # (N, z_dim)
        h_batch = h.expand(1, N, -1).clone() if h is not None else None

        cumulative_reward = torch.zeros(N, device=dev)
        h_after_first     = None

        for step in range(self.horizon):
            actions_t = [seqs[i][step] for i in range(N)]
            enc = encode_action_batch(actions_t, self.characters, self.actor, dev)

            z_pred, h_batch = self.model.step(
                z_batch,
                enc["action_type"], enc["action_char"], enc["action_loc"],
                enc["action_fact"], enc["action_goal"], enc["actor_idx"],
                h_batch,
            )
            z_batch = z_pred.squeeze(1)  # (N, z_dim)

            if step == 0:
                h_after_first = h_batch  # (1, N, gru_hidden) — used to pick h for best

            pred_flat = self.vae.decode(z_batch)         # (N, flat_dim)
            r = compute_reward(pred_flat, self.n_chars)  # (N,)
            cumulative_reward += (self.gamma ** step) * r

        best_idx    = cumulative_reward.argmax().item()
        best_action = seqs[best_idx][0]

        # Slice h_after_first for the winning candidate → (1, 1, gru_hidden)
        h_next = h_after_first[:, best_idx:best_idx + 1, :].contiguous()
        return best_action, h_next
