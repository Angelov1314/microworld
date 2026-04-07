"""
Phase 3: GRU World Model — 5-char 甄嬛传, multi-step rollout.

Pipeline:  state → VAE.encode → z_t → GRU(z_t, a_t, h_{t-1}) → z'_t → VAE.decode → pred_state
VAE is frozen from Phase 1. GRU + head are trained.

Training:  teacher-forced (ground-truth z_t at each step).
Evaluation:
    1-step (teacher-forced)  — measures single-step quality
    5-step open-loop rollout — measures compounding error (key Phase 3 goal)

Success criteria:
    1-step:
        location_acc     >= 99%
        mood_acc         >= 95%
        energy_mae       <= 0.05
        relationship_mae <= 0.05

    5-step open-loop:
        location_acc     >= 97%
        mood_acc         >= 90%
        energy_mae       <= 0.10
        relationship_mae <= 0.10
"""

import sys
sys.path.insert(0, "/Users/jerry/Claude/microworld")

import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data.generator import generate_episodes
from models.world_vae import WorldStateVAE
from models.gru_world_model import GRUWorldModel, SequenceDataset, seq_collate_fn, train
from models.baseline_mlp import component_loss
from core import flat_state_dim
from core.tensorize import CHAR_DIM, rel_dim, LOCATIONS, MOODS


# ─── Episode-level split (keep sequences intact) ──────────────────────────────

def split_episodes_grouped(episodes, test_ratio=0.15, val_ratio=0.05, seed=0):
    rng = random.Random(seed)
    eps = list(episodes)
    rng.shuffle(eps)
    n = len(eps)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    test_eps  = eps[:n_test]
    val_eps   = eps[n_test:n_test + n_val]
    train_eps = eps[n_test + n_val:]
    print(f"Episode split: {len(train_eps)} train / {len(val_eps)} val / {len(test_eps)} test")
    return train_eps, val_eps, test_eps


# ─── 1-step evaluation (teacher-forced) ───────────────────────────────────────

def evaluate_1step(model, vae, dl, n_chars, device):
    """Evaluate every step in every sequence independently (teacher-forced z_t)."""
    n_rels   = rel_dim(n_chars)
    char_end = n_chars * CHAR_DIM
    rel_end  = char_end + n_rels
    le       = len(LOCATIONS)
    me       = le + len(MOODS)

    loc_correct = mood_correct = total = 0
    energy_errs, rel_errs = [], []

    model.eval()
    vae.eval()
    with torch.no_grad():
        for batch in dl:
            s_seq = batch["state_seq"].float().to(device)
            B, T, D = s_seq.shape

            z_flat, _ = vae.encode(s_seq.view(B * T, D))
            z_seq = z_flat.view(B, T, -1)

            z_pred_seq, _ = model(
                z_seq,
                batch["action_type"].to(device),
                batch["action_char"].to(device),
                batch["action_loc"].to(device),
                batch["action_fact"].to(device),
                batch["action_goal"].to(device),
                batch["actor_idx"].to(device),
            )

            pred = vae.decode(z_pred_seq.view(B * T, -1))
            tgt_chars = batch["next_chars"].view(B * T, n_chars, CHAR_DIM).float().to(device)
            tgt_rels  = batch["next_rels"].view(B * T, -1).float().to(device)

            total += B * T * n_chars
            pred_chars = pred[:, :char_end].view(B * T, n_chars, CHAR_DIM)

            loc_correct  += (pred_chars[:, :, :le].argmax(-1) ==
                             tgt_chars[:, :, :le].argmax(-1)).sum().item()
            mood_correct += (pred_chars[:, :, le:me].argmax(-1) ==
                             tgt_chars[:, :, le:me].argmax(-1)).sum().item()

            energy_errs.append(
                np.abs(pred_chars[:, :, me:me+1].cpu().numpy() -
                       tgt_chars[:, :, me:me+1].cpu().numpy()).mean())
            rel_errs.append(torch.abs(pred[:, char_end:rel_end] - tgt_rels).mean().item())

    return {
        "location_acc":     loc_correct / total,
        "mood_acc":         mood_correct / total,
        "energy_mae":       float(np.mean(energy_errs)),
        "relationship_mae": float(np.mean(rel_errs)),
    }


# ─── 5-step open-loop evaluation ──────────────────────────────────────────────

def evaluate_openloop(model, vae, dl, n_chars, n_steps, device):
    """
    Open-loop rollout: feed predicted z' back as input for the next step.
    Evaluates cumulative accuracy after n_steps steps.
    """
    n_rels   = rel_dim(n_chars)
    char_end = n_chars * CHAR_DIM
    rel_end  = char_end + n_rels
    le       = len(LOCATIONS)
    me       = le + len(MOODS)

    # Accumulate per-step metrics
    step_metrics = [{
        "loc_correct": 0, "mood_correct": 0, "total": 0,
        "energy_errs": [], "rel_errs": [],
    } for _ in range(n_steps)]

    model.eval()
    vae.eval()
    with torch.no_grad():
        for batch in dl:
            s_seq = batch["state_seq"].float().to(device)
            B     = s_seq.shape[0]

            # Start from ground-truth z_0
            z = vae.encode(s_seq[:, 0])[0]  # (B, z_dim)
            h = None

            for t in range(n_steps):
                # One GRU step
                z_in = z.unsqueeze(1)  # (B, 1, z_dim)
                z_pred, h = model(
                    z_in,
                    batch["action_type"][:, t:t+1].to(device),
                    batch["action_char"][:, t:t+1].to(device),
                    batch["action_loc"][:, t:t+1].to(device),
                    batch["action_fact"][:, t:t+1].to(device),
                    batch["action_goal"][:, t:t+1].to(device),
                    batch["actor_idx"][:, t:t+1].to(device),
                    h,
                )
                z = z_pred.squeeze(1)  # open-loop: use prediction as next input

                pred      = vae.decode(z)
                tgt_chars = batch["next_chars"][:, t].float().to(device)   # (B, n_chars, CHAR_DIM)
                tgt_rels  = batch["next_rels"][:, t].float().to(device)     # (B, n_rels)

                m = step_metrics[t]
                m["total"]       += B * n_chars
                pred_chars        = pred[:, :char_end].view(B, n_chars, CHAR_DIM)
                m["loc_correct"] += (pred_chars[:, :, :le].argmax(-1) ==
                                     tgt_chars[:, :, :le].argmax(-1)).sum().item()
                m["mood_correct"]+= (pred_chars[:, :, le:me].argmax(-1) ==
                                     tgt_chars[:, :, le:me].argmax(-1)).sum().item()
                m["energy_errs"].append(
                    np.abs(pred_chars[:, :, me:me+1].cpu().numpy() -
                           tgt_chars[:, :, me:me+1].cpu().numpy()).mean())
                m["rel_errs"].append(
                    torch.abs(pred[:, char_end:rel_end] - tgt_rels).mean().item())

    result = []
    for m in step_metrics:
        result.append({
            "location_acc":     m["loc_correct"] / m["total"],
            "mood_acc":         m["mood_correct"] / m["total"],
            "energy_mae":       float(np.mean(m["energy_errs"])),
            "relationship_mae": float(np.mean(m["rel_errs"])),
        })
    return result  # list of 5 metric dicts (one per step)


def check_success(m1, m5):
    one_step_criteria = {
        "location_acc":     (0.99, ">="),
        "mood_acc":         (0.95, ">="),
        "energy_mae":       (0.05, "<="),
        "relationship_mae": (0.05, "<="),
    }
    five_step_criteria = {
        "location_acc":     (0.97, ">="),
        "mood_acc":         (0.90, ">="),
        "energy_mae":       (0.10, "<="),
        "relationship_mae": (0.10, "<="),
    }

    print("\n── 1-Step Success Criteria ─────────────────────────")
    all_pass = True
    for k, (thr, op) in one_step_criteria.items():
        v = m1[k]; passed = (v >= thr) if op == ">=" else (v <= thr)
        if not passed: all_pass = False
        print(f"  {'PASS' if passed else 'FAIL'}  {k}: {v:.4f} (need {op} {thr})")

    print("\n── 5-Step Open-Loop Criteria ───────────────────────")
    for k, (thr, op) in five_step_criteria.items():
        v = m5[k]; passed = (v >= thr) if op == ">=" else (v <= thr)
        if not passed: all_pass = False
        print(f"  {'PASS' if passed else 'FAIL'}  {k}: {v:.4f} (need {op} {thr})")
    print("────────────────────────────────────────────────────")
    return all_pass


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(history, metrics_1step, openloop_steps, out_dir, passed):
    epochs    = list(range(1, len(history["train_loss"]) + 1))
    comp_keys = ["location", "mood", "energy", "relationships"]
    comp_data = {k: [c[k] for c in history["val_components"]] for k in comp_keys}
    BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

    fig = plt.figure(figsize=(18, 10))
    status = "ALL CRITERIA PASSED" if passed else "FAILED"
    fig.suptitle(
        f"Phase 3 — GRU World Model · 5-char 甄嬛传 · {status}",
        fontsize=14, fontweight="bold", color=GREEN if passed else RED, y=0.99,
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    # (A) Loss curves
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(epochs, history["train_loss"], label="Train", color=BLUE, lw=2)
    ax.plot(epochs, history["val_loss"],   label="Val",   color=ORANGE, lw=2)
    ax.set_title("(A) Total Weighted Loss — GRU teacher-forced")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # (B) Per-component val loss
    ax = fig.add_subplot(gs[0, 2])
    for k, c in zip(comp_keys, [BLUE, ORANGE, GREEN, RED]):
        ax.plot(epochs, comp_data[k], label=k, color=c, lw=1.8)
    ax.set_title("(B) Val Loss by Component")
    ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (C) Open-loop accuracy over steps
    ax = fig.add_subplot(gs[0, 3])
    steps = list(range(1, len(openloop_steps) + 1))
    ax.plot(steps, [m["location_acc"] for m in openloop_steps],
            label="location", color=BLUE, lw=2, marker="o")
    ax.plot(steps, [m["mood_acc"] for m in openloop_steps],
            label="mood", color=ORANGE, lw=2, marker="o")
    ax.axhline(0.97, color=BLUE,   linestyle="--", lw=1, alpha=0.6)
    ax.axhline(0.90, color=ORANGE, linestyle="--", lw=1, alpha=0.6)
    ax.set_title("(C) Open-loop Accuracy vs Steps")
    ax.set_xlabel("Rollout Step"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0.7, 1.02)

    # (D) 1-step accuracy bars
    ax = fig.add_subplot(gs[1, 0])
    items = [
        ("location_acc", metrics_1step["location_acc"], 0.99),
        ("mood_acc",     metrics_1step["mood_acc"],      0.95),
    ]
    names   = [x[0] for x in items]
    values  = [x[1] for x in items]
    threshs = [x[2] for x in items]
    bcolors = [GREEN if v >= t else RED for v, t in zip(values, threshs)]
    bars = ax.bar(names, values, color=bcolors, alpha=0.82, zorder=3)
    for bar, t in zip(bars, threshs):
        ax.hlines(t, bar.get_x(), bar.get_x() + bar.get_width(),
                  colors="black", linestyles="--", lw=1.5, zorder=4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylim(0.80, 1.05); ax.set_title("(D) 1-step Accuracy")
    ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3, zorder=0)

    # (E) 1-step MAE bars
    ax = fig.add_subplot(gs[1, 1])
    items_e = [
        ("energy\nMAE",       metrics_1step["energy_mae"],       0.05),
        ("relationship\nMAE", metrics_1step["relationship_mae"],  0.05),
    ]
    names_e   = [x[0] for x in items_e]
    values_e  = [x[1] for x in items_e]
    thresh_e  = [x[2] for x in items_e]
    bcolors_e = [GREEN if v <= t else RED for v, t in zip(values_e, thresh_e)]
    bars_e = ax.bar(names_e, values_e, color=bcolors_e, alpha=0.82, zorder=3)
    for bar, t in zip(bars_e, thresh_e):
        ax.hlines(t, bar.get_x(), bar.get_x() + bar.get_width(),
                  colors="black", linestyles="--", lw=1.5, zorder=4)
    for bar in bars_e:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.07); ax.set_title("(E) 1-step MAE")
    ax.set_ylabel("MAE"); ax.grid(axis="y", alpha=0.3, zorder=0)

    # (F) Open-loop MAE over steps
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(steps, [m["energy_mae"]       for m in openloop_steps],
            label="energy",       color=GREEN,  lw=2, marker="o")
    ax.plot(steps, [m["relationship_mae"] for m in openloop_steps],
            label="relationship", color=RED,    lw=2, marker="o")
    ax.axhline(0.10, color="black", linestyle="--", lw=1, alpha=0.5, label="threshold")
    ax.set_title("(F) Open-loop MAE vs Steps")
    ax.set_xlabel("Rollout Step"); ax.set_ylabel("MAE")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (G) Model summary
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    summary = (
        "GRUWorldModel\n\n"
        "z_dim   : 64\n"
        "GRU     : 108 → 256\n"
        "Head    : 256 → 256 → 64\n"
        "seq_len : 10\n\n"
        "Training: teacher-forced\n"
        "Eval:     5-step open-loop\n\n"
        "Characters:\n"
        "  ZH  HH  ALC\n"
        "  HS  SMZ"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("(G) Model Info")

    out_path = out_dir / "phase3_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device  = "cpu"
    n_chars = 5

    print("=" * 60)
    print("Phase 3: GRU World Model — T(z,a,h)→z'")
    print("Characters: 甄嬛 皇后 安陵容 皇上 沈眉庄")
    print(f"Device: {device}")
    print("=" * 60)

    # ── Load frozen VAE ──
    state_dim = flat_state_dim(n_chars)
    vae = WorldStateVAE(state_dim, z_dim=64, hidden_dim=256).to(device)
    vae_path = Path("experiments/outputs/phase1/best_vae.pt")
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    print(f"\nLoaded VAE from {vae_path}  (z_dim=64, frozen)")

    # ── Generate & split episodes (keep grouped for sequence windows) ──
    print("\n[1/3] Generating episodes...")
    episodes = generate_episodes(
        num_episodes=2000,
        episode_length=50,
        seed=42,
        phase0=False,
    )
    train_eps, val_eps, test_eps = split_episodes_grouped(episodes)

    # ── Train ──
    print("\n[2/3] Training GRUWorldModel (teacher-forced, seq_len=10)...")
    out_dir = Path("experiments/outputs/phase3")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = train(
        episodes_train=train_eps,
        episodes_val=val_eps,
        vae=vae,
        n_chars=n_chars,
        seq_len=10,
        epochs=60,
        batch_size=128,
        lr=1e-3,
        hidden_dim=256,
        gru_hidden=256,
        device=device,
        save_path=str(out_dir / "best_gru.pt"),
    )

    # ── Load best checkpoint ──
    best_path = str(out_dir / "best_gru.pt")
    result["model"].load_state_dict(torch.load(best_path, map_location=device))
    result["model"].eval()
    print(f"Loaded best checkpoint from {best_path}")

    # ── Evaluate ──
    print("\n[3/3] Evaluating on test set...")
    # Need seq_len >= n_steps for open-loop; use seq_len=10
    test_ds = SequenceDataset(test_eps, seq_len=10)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False,
                         collate_fn=seq_collate_fn, num_workers=0)
    print(f"  test sequences: {len(test_ds):,}")

    # 1-step
    m1 = evaluate_1step(result["model"], vae, test_dl, n_chars, device)
    print("\n── 1-Step Test Metrics ──────────────────────────────")
    for k, v in m1.items(): print(f"  {k}: {v:.4f}")

    # 5-step open-loop
    openloop = evaluate_openloop(result["model"], vae, test_dl, n_chars, n_steps=5, device=device)
    print("\n── 5-Step Open-Loop Test Metrics ────────────────────")
    for step, m in enumerate(openloop, 1):
        print(f"  Step {step}: loc={m['location_acc']:.4f}  mood={m['mood_acc']:.4f}  "
              f"energy={m['energy_mae']:.4f}  rel={m['relationship_mae']:.4f}")

    passed = check_success(m1, openloop[-1])  # check against step-5 metrics
    print("\nPhase 3 PASSED. Proceed to Phase 4 (planner)." if passed
          else "\nPhase 3 FAILED. Check GRU architecture or rollout length.")

    import json
    history = result["history"]
    with open(out_dir / "history.json", "w") as f:
        json.dump({
            "train_loss":     [float(x) for x in history["train_loss"]],
            "val_loss":       [float(x) for x in history["val_loss"]],
            "val_components": [{k: float(v) for k, v in c.items()}
                               for c in history["val_components"]],
            "tf_ratio":       [float(x) for x in history["tf_ratio"]],
            "1step_metrics":  {k: float(v) for k, v in m1.items()},
            "openloop_steps": [{k: float(v) for k, v in m.items()} for m in openloop],
        }, f, indent=2)

    plot_results(history, m1, openloop, out_dir, passed)
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
