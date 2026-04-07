"""
MicroWorld Narrative Demo

Runs the GRU world model + CEM planner and prints a human-readable
story of what happens. Shows:
  1. What the planner imagines will happen (world-model rollout)
  2. What actually happens (true engine)
  3. How well the model predicted it

Usage:
    cd /Users/jerry/Claude/microworld
    python3 demo/narrative_demo.py [--steps 20] [--actor 甄嬛] [--seed 99]
    python3 demo/narrative_demo.py --random   # random policy baseline
"""

import sys
sys.path.insert(0, "/Users/jerry/Claude/microworld")

import argparse
import random
import torch
import numpy as np
from pathlib import Path

from core.world import make_random_world, CHARACTERS_FULL, MOODS, LOCATIONS
from core.engine import transition, apply_stochastic_events
from core import flat_state_dim, encode_state, encode_action
from core.tensorize import CHAR_DIM, rel_dim, LOCATIONS as LOCS
from models.world_vae import WorldStateVAE
from models.gru_world_model import GRUWorldModel
from models.cem_planner import CEMPlanner
from models.planner import sample_random_action, MOOD_SCORES


# ─── Narrative helpers ────────────────────────────────────────────────────────

CHAR_DISPLAY = {
    "甄嬛":   "甄嬛",
    "皇后":   "皇后",
    "安陵容":  "安陵容",
    "皇上":   "皇上",
    "沈眉庄":  "沈眉庄",
}

MOOD_EMOJI = {
    "calm":    "😌",
    "anxious": "😰",
    "angry":   "😡",
    "happy":   "😊",
    "sad":     "😢",
}

ACTION_NARRATIVE = {
    "move_to":   lambda a: f"移驾至【{a.target_loc}】",
    "talk_to":   lambda a: f"与{a.target_char}交谈",
    "argue":     lambda a: f"与{a.target_char}发生争执",
    "cooperate": lambda a: f"与{a.target_char}合作",
    "rest":      lambda a: "在寝宫休息养神",
    "inform":    lambda a: f"向{a.target_char}透露【{a.target_fact}】",
    "plan":      lambda a: f"暗定计策：{a.target_goal}",
    "observe":   lambda a: "暗中观察四周",
    "wait":      lambda a: "静待时机",
    "react":     lambda a: f"应对事件【{a.target_event or ''}】",
}

def describe_action(action) -> str:
    fn = ACTION_NARRATIVE.get(action.type, lambda a: a.type)
    return fn(action)

def mood_bar(energy: float) -> str:
    filled = int(energy * 10)
    return "█" * filled + "░" * (10 - filled)

def print_state_summary(state, characters, actor, step=None):
    char = state.characters[actor]
    emoji = MOOD_EMOJI.get(char.mood, "")
    header = f"  Step {step:2d}" if step is not None else "  初始状态"
    print(f"{header} │ {actor} @ 【{char.location}】{emoji}{char.mood:8s} "
          f"精力[{mood_bar(char.energy)}]{char.energy:.2f} 目标:{char.current_goal}")

    # Show key relationships
    others = [c for c in characters if c != actor]
    rels = [(c, state.get_rel(actor, c)) for c in others]
    rel_str = "  ".join(f"{c}:{v:+.2f}" for c, v in rels)
    print(f"         │ 关系: {rel_str}")

def print_world_snapshot(state, characters):
    """Print compact world state for all characters."""
    for name in characters:
        char = state.characters[name]
        emoji = MOOD_EMOJI.get(char.mood, "")
        bar = mood_bar(char.energy)
        print(f"    {name:5s}│ {char.location:5s} {emoji} [{bar}]{char.energy:.2f}")


def compute_prediction_accuracy(pred_state_tensor, true_next_state,
                                  characters, n_chars):
    """Compare model's predicted next state vs true next state."""
    from core.tensorize import CHAR_DIM, LOCATIONS, MOODS
    le = len(LOCATIONS)
    me = le + len(MOODS)
    char_end = n_chars * CHAR_DIM

    pred_chars = pred_state_tensor[:char_end].view(n_chars, CHAR_DIM)
    results = {}

    for i, name in enumerate(characters):
        true_char = true_next_state.characters[name]
        pred_loc  = LOCATIONS[pred_chars[i, :le].argmax().item()]
        pred_mood = MOODS[pred_chars[i, le:me].argmax().item()]
        pred_energy = pred_chars[i, me].item()

        results[name] = {
            "loc_correct":    pred_loc  == true_char.location,
            "mood_correct":   pred_mood == true_char.mood,
            "energy_error":   abs(pred_energy - true_char.energy),
            "pred_loc":       pred_loc,
            "pred_mood":      pred_mood,
        }
    return results


# ─── Main demo ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int, default=20)
    parser.add_argument("--actor",  type=str, default="甄嬛")
    parser.add_argument("--seed",   type=int, default=99)
    parser.add_argument("--random", action="store_true",
                        help="Use random policy instead of CEM planner")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--horizon",   type=int, default=5)
    args = parser.parse_args()

    device  = "cpu"
    n_chars = 5
    chars   = CHARACTERS_FULL
    actor   = args.actor

    print("=" * 70)
    print("  MicroWorld 甄嬛传 叙事演示")
    print(f"  控制角色: {actor}  {'随机策略' if args.random else f'CEM规划 (N={args.n_samples}, H={args.horizon})'}")
    print(f"  回合数: {args.steps}  随机种子: {args.seed}")
    print("=" * 70)

    # ── Load models ──
    state_dim = flat_state_dim(n_chars)
    vae = WorldStateVAE(state_dim, z_dim=64, hidden_dim=256).to(device)
    vae.load_state_dict(torch.load(
        Path("experiments/outputs/phase1/best_vae.pt"), map_location=device))
    vae.eval()

    gru = GRUWorldModel(z_dim=64, n_chars=n_chars, hidden_dim=256, gru_hidden=256).to(device)
    gru.load_state_dict(torch.load(
        Path("experiments/outputs/phase3/best_gru.pt"), map_location=device))
    gru.eval()

    planner = CEMPlanner(
        gru_model=gru, vae=vae, n_chars=n_chars,
        n_samples=args.n_samples, horizon=args.horizon,
        n_iters=3, elite_frac=0.1,
        actor=actor, characters=chars,
        device=device, seed=args.seed,
    )

    # ── Initialize world ──
    rng = random.Random(args.seed)
    state = make_random_world(characters=chars, seed=args.seed)

    s_enc  = encode_state(state, chars)
    z, _   = vae.encode(s_enc["flat"].float().unsqueeze(0))
    h      = None

    print("\n  ─── 初始宫廷状态 ──────────────────────────────────────────────")
    print_world_snapshot(state, chars)
    print()

    # ── Tracking ──
    step_rewards   = []
    loc_correct    = mood_correct = total_preds = 0
    energy_errors  = []

    print("  ─── 故事开始 ──────────────────────────────────────────────────\n")

    for step in range(1, args.steps + 1):
        # Plan
        if args.random:
            action = sample_random_action(actor, chars, rng)
            h_next = h  # random policy doesn't update h
        else:
            action, h_next = planner.plan(z.squeeze(0), h)

        # Imagined next state (from world model)
        with torch.no_grad():
            a_enc     = encode_action(action, chars)
            actor_idx = torch.tensor([chars.index(actor)], dtype=torch.long)
            # z is (1, z_dim); step expects (B, z_dim) where B=1
            z_pred, _ = gru.step(
                z,                               # (1, z_dim)
                a_enc["type"].unsqueeze(0),      # (1,)
                a_enc["target_char"].unsqueeze(0),
                a_enc["target_loc"].unsqueeze(0),
                a_enc["target_fact"].unsqueeze(0),
                a_enc["target_goal"].unsqueeze(0),
                actor_idx,                       # (1,)
                h,
            )
            imagined_flat = vae.decode(z_pred.squeeze(1)).squeeze(0)  # (state_dim,)

        # True transition
        true_next = transition(state, action)

        # Compute reward from true state
        char     = true_next.characters[actor]
        mood_idx = MOODS.index(char.mood)
        r = 0.5 * MOOD_SCORES[mood_idx].item() + 0.3 * char.energy + \
            0.2 * (state.get_rel(actor, "皇上") + 1.0) / 2.0
        step_rewards.append(r)

        # Prediction accuracy
        acc = compute_prediction_accuracy(
            imagined_flat, true_next, chars, n_chars)
        for name in chars:
            total_preds   += 1
            loc_correct   += int(acc[name]["loc_correct"])
            mood_correct  += int(acc[name]["mood_correct"])
            energy_errors.append(acc[name]["energy_error"])

        # Print narrative
        action_text = describe_action(action)
        print(f"  ┌ Step {step:2d} ─────────────────────────────────────────────────")
        print(f"  │ {actor} {action_text}")
        print(f"  │ 模型预测 → 位置{'✓' if acc[actor]['loc_correct'] else '✗'}"
              f"({acc[actor]['pred_loc']}) "
              f"心情{'✓' if acc[actor]['mood_correct'] else '✗'}"
              f"({acc[actor]['pred_mood']})")
        print_state_summary(true_next, chars, actor, step)
        print(f"  └ 奖励: {r:.3f}  本步精力: {char.energy:.2f}\n")

        # Apply stochastic events, update state and latent
        next_state = apply_stochastic_events(true_next)
        if next_state.active_events != true_next.active_events:
            new_evt = next_state.active_events - true_next.active_events
            print(f"  ⚡ 宫廷事件: {', '.join(new_evt)}\n")

        state = next_state
        ns_enc = encode_state(state, chars)
        z, _   = vae.encode(ns_enc["flat"].float().unsqueeze(0))
        if not args.random:
            h = h_next

    # ── Final summary ──
    mean_reward = np.mean(step_rewards)
    print("=" * 70)
    print(f"  ─── 终局总结 ─────────────────────────────────────────────────")
    print(f"  {actor} 最终状态:")
    final_char = state.characters[actor]
    print(f"    位置: {final_char.location}")
    print(f"    心情: {MOOD_EMOJI.get(final_char.mood, '')} {final_char.mood}")
    print(f"    精力: [{mood_bar(final_char.energy)}] {final_char.energy:.2f}")
    print(f"    目标: {final_char.current_goal}")
    print(f"    知识: {', '.join(final_char.knowledge) if final_char.knowledge else '无'}")
    print()
    print(f"  ─── 模型预测准确率 ──────────────────────────────────────────")
    print(f"    位置准确率:  {loc_correct/total_preds:.1%}  ({loc_correct}/{total_preds})")
    print(f"    心情准确率:  {mood_correct/total_preds:.1%}  ({mood_correct}/{total_preds})")
    print(f"    精力 MAE:   {np.mean(energy_errors):.4f}")
    print()
    print(f"  ─── 规划表现 ────────────────────────────────────────────────")
    print(f"    平均奖励: {mean_reward:.4f}")
    print(f"    累计奖励: {sum(step_rewards):.3f}")
    print(f"    奖励趋势: {np.polyfit(range(len(step_rewards)), step_rewards, 1)[0]:+.4f}/step")
    print("=" * 70)


if __name__ == "__main__":
    main()
