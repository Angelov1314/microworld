"""
MicroWorld: State <-> Tensor conversion.

Maps structured WorldState to flat tensors and back.
Handles: one-hot categoricals, multi-hot sets, continuous floats, directed relationships.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Optional

from .world import (
    WorldState, CharacterState, Action,
    LOCATIONS, MOODS, GOALS, FACTS, EVENTS,
    ACTION_TYPES, CHARACTERS_FULL
)

# ─── Index maps ───────────────────────────────────────────────────────────────

LOC2I   = {v: i for i, v in enumerate(LOCATIONS)}
MOOD2I  = {v: i for i, v in enumerate(MOODS)}
GOAL2I  = {v: i for i, v in enumerate(GOALS)}
FACT2I  = {v: i for i, v in enumerate(FACTS)}
EVENT2I = {v: i for i, v in enumerate(EVENTS)}
ACT2I   = {v: i for i, v in enumerate(ACTION_TYPES)}
NULL_IDX = -1  # padding for optional fields

# ─── Character feature dims ────────────────────────────────────────────────────
# location:  one-hot  len(LOCATIONS) = 3
# mood:      one-hot  len(MOODS)     = 5
# energy:    scalar   1
# goal:      one-hot  len(GOALS)     = 10
# knowledge: multi-hot len(FACTS)    = 20
# TOTAL per character = 3 + 5 + 1 + 10 + 20 = 39

CHAR_DIM = len(LOCATIONS) + len(MOODS) + 1 + len(GOALS) + len(FACTS)  # 39

# ─── Environment dims ─────────────────────────────────────────────────────────
# time_of_day: one-hot  4
# weather:     one-hot  3
# active_events: multi-hot 20
# TOTAL = 4 + 3 + 20 = 27

TIME_OF_DAY = ["morning", "afternoon", "evening", "night"]
WEATHER     = ["sunny", "rainy", "cloudy"]
TIME2I  = {v: i for i, v in enumerate(TIME_OF_DAY)}
WEATH2I = {v: i for i, v in enumerate(WEATHER)}
ENV_DIM = len(TIME_OF_DAY) + len(WEATHER) + len(EVENTS)  # 27


def encode_character(char: CharacterState) -> torch.Tensor:
    """Encode one CharacterState -> float tensor of shape (CHAR_DIM,)."""
    loc = torch.zeros(len(LOCATIONS))
    loc[LOC2I[char.location]] = 1.0

    mood = torch.zeros(len(MOODS))
    mood[MOOD2I[char.mood]] = 1.0

    energy = torch.tensor([char.energy], dtype=torch.float32)

    goal = torch.zeros(len(GOALS))
    goal[GOAL2I[char.current_goal]] = 1.0

    knowledge = torch.zeros(len(FACTS))
    for fact in char.knowledge:
        if fact in FACT2I:
            knowledge[FACT2I[fact]] = 1.0

    return torch.cat([loc, mood, energy, goal, knowledge])  # (39,)


def decode_character(vec: torch.Tensor) -> dict:
    """Decode float tensor (39,) -> dict of predicted values (with logits)."""
    offset = 0

    loc_logits = vec[offset: offset + len(LOCATIONS)]; offset += len(LOCATIONS)
    mood_logits = vec[offset: offset + len(MOODS)]; offset += len(MOODS)
    energy = vec[offset].item(); offset += 1
    goal_logits = vec[offset: offset + len(GOALS)]; offset += len(GOALS)
    knowledge_logits = vec[offset: offset + len(FACTS)]

    return {
        "location": LOCATIONS[loc_logits.argmax().item()],
        "mood": MOODS[mood_logits.argmax().item()],
        "energy": float(np.clip(energy, 0.0, 1.0)),
        "goal": GOALS[goal_logits.argmax().item()],
        "knowledge": frozenset(
            FACTS[i] for i in range(len(FACTS)) if knowledge_logits[i].item() > 0.5
        ),
        # raw logits for loss computation
        "loc_logits": loc_logits,
        "mood_logits": mood_logits,
        "goal_logits": goal_logits,
        "knowledge_logits": knowledge_logits,
    }


def encode_environment(state: WorldState) -> torch.Tensor:
    """Encode environment variables -> float tensor of shape (ENV_DIM,)."""
    time = torch.zeros(len(TIME_OF_DAY))
    time[TIME2I.get(state.time_of_day, 0)] = 1.0

    weather = torch.zeros(len(WEATHER))
    weather[WEATH2I.get(state.weather, 0)] = 1.0

    events = torch.zeros(len(EVENTS))
    for evt in state.active_events:
        if evt in EVENT2I:
            events[EVENT2I[evt]] = 1.0

    return torch.cat([time, weather, events])  # (27,)


def encode_relationships(state: WorldState, characters: list[str]) -> torch.Tensor:
    """
    Encode directed relationship matrix.
    Returns float tensor of shape (n*(n-1),) — directed pairs only (no self-loops).
    """
    pairs = [(a, b) for a in characters for b in characters if a != b]
    rels = torch.zeros(len(pairs))
    for i, (a, b) in enumerate(pairs):
        rels[i] = state.get_rel(a, b)
    return rels  # values in [-1, 1]


def encode_state(state: WorldState, characters: Optional[list[str]] = None) -> dict:
    """
    Full WorldState -> tensor dict.
    Returns separate tensors per component for flexible loss computation.
    """
    if characters is None:
        characters = sorted(state.characters.keys())

    char_tensors = torch.stack([encode_character(state.characters[c]) for c in characters])
    # shape: (n_chars, CHAR_DIM)

    rel_tensor = encode_relationships(state, characters)
    # shape: (n*(n-1),)

    env_tensor = encode_environment(state)
    # shape: (ENV_DIM,)

    # Flat tensor for direct MLP (Phase 0 baseline)
    flat = torch.cat([char_tensors.flatten(), rel_tensor, env_tensor])

    return {
        "characters": char_tensors,       # (n, 39)
        "relationships": rel_tensor,       # (n*(n-1),)
        "environment": env_tensor,         # (27,)
        "flat": flat,                      # everything concatenated
        "character_order": characters,
    }


def encode_action(action: Action, characters: list[str]) -> dict:
    """
    Encode parameterized action -> tensor dict.
    Uses separate embeddings for each field (type, target_char, target_loc, target_fact, target_goal).
    NULL is encoded as index 0 in each embedding (reserved).
    """
    CHAR2I = {c: i + 1 for i, c in enumerate(characters)}  # 0 = null
    LOC2I_WITH_NULL = {loc: i + 1 for i, loc in enumerate(LOCATIONS)}
    FACT2I_WITH_NULL = {f: i + 1 for i, f in enumerate(FACTS)}
    GOAL2I_WITH_NULL = {g: i + 1 for i, g in enumerate(GOALS)}

    return {
        "type":        torch.tensor(ACT2I.get(action.type, 0), dtype=torch.long),
        "target_char": torch.tensor(CHAR2I.get(action.target_char, 0), dtype=torch.long),
        "target_loc":  torch.tensor(LOC2I_WITH_NULL.get(action.target_loc, 0), dtype=torch.long),
        "target_fact": torch.tensor(FACT2I_WITH_NULL.get(action.target_fact, 0), dtype=torch.long),
        "target_goal": torch.tensor(GOAL2I_WITH_NULL.get(action.target_goal, 0), dtype=torch.long),
    }


# ─── Flat state dim calculator ────────────────────────────────────────────────

def flat_state_dim(n_chars: int) -> int:
    n_rels = n_chars * (n_chars - 1)
    return n_chars * CHAR_DIM + n_rels + ENV_DIM


def rel_dim(n_chars: int) -> int:
    return n_chars * (n_chars - 1)
