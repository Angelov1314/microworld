"""
MicroWorld: Dataset Generator.

Generates (state, action, next_state) transitions from the deterministic engine.
Stochastic events applied AFTER recording the transition (model not trained on them).
Split by EPISODE (not transition) to prevent data leakage.
"""

from __future__ import annotations
import random
import pickle
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from core.world import make_random_world, Action
from core.engine import transition, apply_stochastic_events, valid_actions


def generate_episodes(
    num_episodes: int = 1000,
    episode_length: int = 50,
    characters: Optional[list[str]] = None,
    seed: Optional[int] = 42,
    phase0: bool = False,
) -> list[list[dict]]:
    """
    Generate episodes of (state, action, next_state) transitions.

    Args:
        num_episodes: number of independent episodes
        episode_length: steps per episode
        characters: character names (default: 5)
        seed: random seed for reproducibility
        phase0: if True, use 2-character tiny world

    Returns:
        list of episodes, each episode is a list of transition dicts
    """
    if seed is not None:
        random.seed(seed)

    if phase0:
        from core.world import CHARACTERS_PHASE0
        characters = CHARACTERS_PHASE0
    elif characters is None:
        from core.world import CHARACTERS_FULL
        characters = CHARACTERS_FULL

    episodes = []

    for ep_idx in tqdm(range(num_episodes), desc="Generating episodes"):
        episode = []
        ep_seed = seed + ep_idx if seed is not None else None
        state = make_random_world(characters=characters, seed=ep_seed)

        for _ in range(episode_length):
            # Pick random actor
            actor = random.choice(characters)

            # Pick random valid action
            acts = valid_actions(state, actor)
            if not acts:
                # No valid actions: wait
                action = Action(type="wait", actor=actor)
            else:
                action = random.choice(acts)

            # Deterministic transition (what the model learns)
            next_state = transition(state, action)

            episode.append({
                "state": state,
                "actor": actor,
                "action": action,
                "next_state": next_state,
                "characters": characters,
            })

            # Apply stochastic events AFTER recording
            # (model is NOT trained to predict these)
            next_state = apply_stochastic_events(next_state)
            state = next_state

        episodes.append(episode)

    return episodes


def split_episodes(
    episodes: list,
    test_ratio: float = 0.15,
    val_ratio: float = 0.05,
    seed: int = 0,
) -> tuple[list, list, list]:
    """
    Split by EPISODE (not transition) to prevent data leakage.
    Returns (train_transitions, val_transitions, test_transitions).
    """
    rng = random.Random(seed)
    eps = list(episodes)
    rng.shuffle(eps)

    n = len(eps)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_eps  = eps[:n_test]
    val_eps   = eps[n_test: n_test + n_val]
    train_eps = eps[n_test + n_val:]

    train = [t for ep in train_eps for t in ep]
    val   = [t for ep in val_eps   for t in ep]
    test  = [t for ep in test_eps  for t in ep]

    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test transitions")
    print(f"       ({len(train_eps)} / {len(val_eps)} / {len(test_eps)} episodes)")
    return train, val, test


def save_dataset(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved dataset to {path}")


def load_dataset(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0", action="store_true", help="Generate tiny 2-char world")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/dataset.pkl")
    args = parser.parse_args()

    print(f"Generating {'Phase 0 (2-char)' if args.phase0 else 'Full (5-char)'} dataset...")
    episodes = generate_episodes(
        num_episodes=args.episodes,
        episode_length=args.length,
        seed=args.seed,
        phase0=args.phase0,
    )
    train, val, test = split_episodes(episodes)

    save_dataset({
        "train": train,
        "val": val,
        "test": test,
        "config": vars(args),
    }, args.out)
