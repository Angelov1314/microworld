"""
Phase 4: Random Shooting Planner — 5-char 甄嬛传 world.

Protocol:
    - 30 independent episodes, each 20 planning steps
    - At each step the PLANNER picks an action via random shooting (N=200, H=5)
    - At each step the RANDOM policy picks a uniformly random action
    - Both policies execute in the TRUE deterministic engine (not world model)
    - Reward is computed from TRUE next state

Success criteria:
    - Planner mean reward > random mean reward  (paired t-test p < 0.05)
    - Planner mean reward >= random mean reward * 1.15  (at least 15% better)
    - 甄嬛 avg energy    >= 0.35  (planner preserves energy)
    - 甄嬛 avg mood_score >= 0.50  (planner maintains positive mood)
"""

import sys
sys.path.insert(0, "/Users/jerry/Claude/microworld")

import random
import torch
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data.generator import generate_episodes
from models.world_vae import WorldStateVAE
from models.gru_world_model import GRUWorldModel
from models.planner import RandomShootingPlanner, compute_reward, sample_random_action, MOOD_SCORES
from core import flat_state_dim, encode_state, encode_action
from core.tensorize import CHAR_DIM, LOCATIONS, MOODS
from core.world import make_random_world, CHARACTERS_FULL
from core.engine import transition, valid_actions


# ─── True-state reward ────────────────────────────────────────────────────────

def true_state_reward(state, actor="甄嬛", emperor="皇上") -> float:
    """Compute reward from ground-truth WorldState."""
    char = state.characters[actor]
    mood_idx = MOODS.index(char.mood)
    mood_score = MOOD_SCORES[mood_idx].item()
    energy = char.energy
    rel_emperor = (state.get_rel(actor, emperor) + 1.0) / 2.0
    return 0.5 * mood_score + 0.3 * energy + 0.2 * rel_emperor


# ─── Single-episode evaluation ────────────────────────────────────────────────

def run_episode(planner, vae, n_steps, chars, actor, seed, use_planner=True):
    """
    Run one episode with either the planner or random policy.
    Returns (rewards, energies, mood_scores) lists of length n_steps.
    """
    rng = random.Random(seed)
    state = make_random_world(characters=chars, seed=seed)

    # Encode initial state
    s_enc = encode_state(state, chars)
    z, _  = vae.encode(s_enc["flat"].float().unsqueeze(0))
    h     = None  # GRU hidden state

    rewards, energies, mood_scores = [], [], []

    for step in range(n_steps):
        if use_planner:
            action, h_next = planner.plan(z.squeeze(0), h)
        else:
            # Random policy: sample random action for actor
            action = sample_random_action(actor, chars, rng)

        # Execute in TRUE engine
        next_state = transition(state, action)

        # Compute reward from true state
        r = true_state_reward(next_state, actor)
        rewards.append(r)
        energies.append(next_state.characters[actor].energy)
        mood_scores.append(MOOD_SCORES[MOODS.index(next_state.characters[actor].mood)].item())

        # Update z and h from true next state
        ns_enc = encode_state(next_state, chars)
        z, _   = vae.encode(ns_enc["flat"].float().unsqueeze(0))
        if use_planner:
            h = h_next  # carry GRU context
        state  = next_state

    return rewards, energies, mood_scores


def check_success(planner_rewards, random_rewards, planner_energy, planner_mood):
    planner_mean = np.mean(planner_rewards)
    random_mean  = np.mean(random_rewards)
    ratio        = planner_mean / (random_mean + 1e-8)
    t_stat, pval = stats.ttest_rel(planner_rewards, random_rewards)

    avg_energy = np.mean(planner_energy)
    avg_mood   = np.mean(planner_mood)

    criteria = [
        ("planner > random (p < 0.05)", pval < 0.05 and t_stat > 0, f"t={t_stat:.3f} p={pval:.4f}"),
        ("planner >= random × 1.15",    ratio >= 1.15,               f"{planner_mean:.4f} vs {random_mean:.4f} ({ratio:.2f}x)"),
        ("甄嬛 avg energy >= 0.35",      avg_energy >= 0.35,          f"{avg_energy:.4f}"),
        ("甄嬛 avg mood_score >= 0.50",  avg_mood   >= 0.50,          f"{avg_mood:.4f}"),
    ]

    print("\n── Success Criteria Check ──────────────────────────")
    all_pass = True
    for name, passed, detail in criteria:
        if not passed: all_pass = False
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")
    print("────────────────────────────────────────────────────")
    return all_pass, planner_mean, random_mean


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(step_rewards_p, step_rewards_r, all_energies, all_moods,
                 planner_mean, random_mean, out_dir, passed):
    BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"
    n_steps = len(step_rewards_p[0])
    steps   = list(range(1, n_steps + 1))

    fig = plt.figure(figsize=(16, 9))
    status = "ALL CRITERIA PASSED" if passed else "FAILED"
    fig.suptitle(
        f"Phase 4 — Random Shooting Planner · 5-char 甄嬛传 · {status}",
        fontsize=14, fontweight="bold", color=GREEN if passed else RED, y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # (A) Per-step mean reward
    ax = fig.add_subplot(gs[0, :2])
    p_mean = np.mean(step_rewards_p, axis=0)
    r_mean = np.mean(step_rewards_r, axis=0)
    p_std  = np.std(step_rewards_p, axis=0)
    r_std  = np.std(step_rewards_r, axis=0)
    ax.plot(steps, p_mean, label="Planner", color=BLUE, lw=2)
    ax.fill_between(steps, p_mean - p_std, p_mean + p_std, color=BLUE, alpha=0.15)
    ax.plot(steps, r_mean, label="Random",  color=ORANGE, lw=2)
    ax.fill_between(steps, r_mean - r_std, r_mean + r_std, color=ORANGE, alpha=0.15)
    ax.set_title("(A) Step Reward — Planner vs Random  (mean ± 1 std)")
    ax.set_xlabel("Planning Step"); ax.set_ylabel("Reward")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)

    # (B) Cumulative reward comparison
    ax = fig.add_subplot(gs[0, 2])
    cum_p = np.cumsum(step_rewards_p, axis=1)
    cum_r = np.cumsum(step_rewards_r, axis=1)
    ax.plot(steps, cum_p.mean(0), label="Planner", color=BLUE, lw=2)
    ax.plot(steps, cum_r.mean(0), label="Random",  color=ORANGE, lw=2)
    ax.set_title("(B) Cumulative Reward")
    ax.set_xlabel("Step"); ax.set_ylabel("Cumulative Reward")
    ax.legend(); ax.grid(alpha=0.3)

    # (C) 甄嬛 energy over time
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps, np.mean(all_energies, axis=0), color=GREEN, lw=2, label="Planner")
    ax.fill_between(steps,
                    np.mean(all_energies, axis=0) - np.std(all_energies, axis=0),
                    np.mean(all_energies, axis=0) + np.std(all_energies, axis=0),
                    color=GREEN, alpha=0.2)
    ax.axhline(0.35, color="black", linestyle="--", lw=1.5, label="threshold")
    ax.set_title("(C) 甄嬛 Energy (Planner)")
    ax.set_xlabel("Step"); ax.set_ylabel("Energy")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)

    # (D) 甄嬛 mood score over time
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(steps, np.mean(all_moods, axis=0), color=BLUE, lw=2)
    ax.fill_between(steps,
                    np.mean(all_moods, axis=0) - np.std(all_moods, axis=0),
                    np.mean(all_moods, axis=0) + np.std(all_moods, axis=0),
                    color=BLUE, alpha=0.2)
    ax.axhline(0.50, color="black", linestyle="--", lw=1.5, label="threshold")
    ax.set_title("(D) 甄嬛 Mood Score (Planner)")
    ax.set_xlabel("Step"); ax.set_ylabel("Mood Score (0-1)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)

    # (E) Summary bar chart
    ax = fig.add_subplot(gs[1, 2])
    labels = ["Planner\nmean reward", "Random\nmean reward"]
    values = [planner_mean, random_mean]
    colors = [GREEN if planner_mean > random_mean else RED, ORANGE]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, zorder=3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_title("(E) Mean Reward Comparison")
    ax.set_ylabel("Mean Step Reward"); ax.grid(axis="y", alpha=0.3, zorder=0)

    out_path = out_dir / "phase4_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device  = "cpu"
    n_chars = 5
    chars   = CHARACTERS_FULL
    actor   = "甄嬛"

    print("=" * 60)
    print("Phase 4: Random Shooting Planner")
    print("Characters: 甄嬛 皇后 安陵容 皇上 沈眉庄")
    print(f"Controlling: {actor}  |  N=200 candidates, H=5 horizon")
    print(f"Device: {device}")
    print("=" * 60)

    # ── Load frozen VAE ──
    state_dim = flat_state_dim(n_chars)
    vae = WorldStateVAE(state_dim, z_dim=64, hidden_dim=256).to(device)
    vae_path = Path("experiments/outputs/phase1/best_vae.pt")
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    print(f"Loaded VAE from {vae_path}")

    # ── Load GRU world model ──
    gru = GRUWorldModel(z_dim=64, n_chars=n_chars, hidden_dim=256, gru_hidden=256).to(device)
    gru_path = Path("experiments/outputs/phase3/best_gru.pt")
    gru.load_state_dict(torch.load(gru_path, map_location=device))
    gru.eval()
    print(f"Loaded GRU from {gru_path}")

    # ── Build planner ──
    planner = RandomShootingPlanner(
        gru_model=gru,
        vae=vae,
        n_chars=n_chars,
        n_samples=200,
        horizon=5,
        gamma=0.95,
        actor=actor,
        characters=chars,
        device=device,
        seed=42,
    )

    # ── Run evaluation ──
    n_episodes = 30
    n_steps    = 20
    out_dir    = Path("experiments/outputs/phase4")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/2] Running {n_episodes} episodes × {n_steps} steps...")
    print("      Planner (random shooting) vs Random policy")

    step_rewards_p = []  # (n_episodes, n_steps)
    step_rewards_r = []
    all_energies   = []
    all_moods      = []

    for ep in range(n_episodes):
        seed = 1000 + ep
        rp, ep_energy, ep_mood = run_episode(
            planner, vae, n_steps, chars, actor, seed=seed, use_planner=True)
        rr, _, _ = run_episode(
            planner, vae, n_steps, chars, actor, seed=seed, use_planner=False)

        step_rewards_p.append(rp)
        step_rewards_r.append(rr)
        all_energies.append(ep_energy)
        all_moods.append(ep_mood)

        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1:2d}/{n_episodes} | "
                  f"planner_mean={np.mean(rp):.4f} random_mean={np.mean(rr):.4f}")

    step_rewards_p = np.array(step_rewards_p)  # (30, 20)
    step_rewards_r = np.array(step_rewards_r)
    all_energies   = np.array(all_energies)
    all_moods      = np.array(all_moods)

    # Flatten for statistical test (per-step rewards across all episodes)
    flat_p = step_rewards_p.flatten()
    flat_r = step_rewards_r.flatten()

    print(f"\n── Summary ──────────────────────────────────────────")
    print(f"  Planner mean reward: {np.mean(flat_p):.4f} ± {np.std(flat_p):.4f}")
    print(f"  Random  mean reward: {np.mean(flat_r):.4f} ± {np.std(flat_r):.4f}")
    print(f"  Ratio:               {np.mean(flat_p)/np.mean(flat_r):.3f}x")

    print("\n[2/2] Checking success criteria...")
    passed, planner_mean, random_mean = check_success(
        flat_p, flat_r,
        all_energies.flatten(),
        all_moods.flatten(),
    )

    print("\nPhase 4 PASSED. World model pipeline complete!" if passed
          else "\nPhase 4 FAILED. Try more samples or longer horizon.")

    import json
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "planner_mean":  float(np.mean(flat_p)),
            "random_mean":   float(np.mean(flat_r)),
            "ratio":         float(np.mean(flat_p) / np.mean(flat_r)),
            "avg_energy":    float(np.mean(all_energies)),
            "avg_mood":      float(np.mean(all_moods)),
            "step_rewards_planner": step_rewards_p.tolist(),
            "step_rewards_random":  step_rewards_r.tolist(),
        }, f, indent=2)

    plot_results(step_rewards_p, step_rewards_r, all_energies, all_moods,
                 planner_mean, random_mean, out_dir, passed)
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
