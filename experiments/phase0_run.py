"""
Phase 0 Experiment: Validate the full pipeline on the minimal 2-character world.

Run:
    cd /Users/jerry/Claude/microworld
    python experiments/phase0_run.py

Success criteria:
    location accuracy  > 99%
    mood accuracy      > 95%
    energy MAE         < 0.05
    relationship MAE   < 0.05
"""

import sys
sys.path.insert(0, "/Users/jerry/Claude/microworld")

import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data.generator import generate_episodes, split_episodes
from models.baseline_mlp import train, TransitionDataset, collate_fn, component_loss
from core import flat_state_dim
from core.tensorize import LOCATIONS, MOODS

def evaluate_accuracy(model, val_dl, n_chars, device):
    """Compute per-component accuracy/error metrics."""
    from core.tensorize import CHAR_DIM, rel_dim, GOALS, FACTS
    n_rels = rel_dim(n_chars)
    char_end = n_chars * CHAR_DIM
    rel_end  = char_end + n_rels

    loc_correct = mood_correct = goal_correct = total = 0
    energy_errs = []
    rel_errs = []
    know_tp = know_fp = know_fn = 0

    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            s   = batch["state_flat"].float().to(device)
            at  = batch["action_type"].to(device)
            ac  = batch["action_char"].to(device)
            al  = batch["action_loc"].to(device)
            af  = batch["action_fact"].to(device)
            ag  = batch["action_goal"].to(device)
            ai  = batch["actor_idx"].to(device)

            pred = model(s, at, ac, al, af, ag, ai)
            B = pred.shape[0]
            total += B * n_chars

            pred_chars = pred[:, :char_end].view(B, n_chars, CHAR_DIM)
            tgt_chars  = batch["next_chars"].float().to(device)
            pred_rels  = pred[:, char_end:rel_end]
            tgt_rels   = batch["next_rels"].float().to(device)

            # Location accuracy
            pred_loc = pred_chars[:, :, :len(LOCATIONS)].argmax(-1)
            tgt_loc  = tgt_chars[:, :, :len(LOCATIONS)].argmax(-1)
            loc_correct += (pred_loc == tgt_loc).sum().item()

            # Mood accuracy
            le = len(LOCATIONS)
            me = le + len(MOODS)
            pred_mood = pred_chars[:, :, le:me].argmax(-1)
            tgt_mood  = tgt_chars[:, :, le:me].argmax(-1)
            mood_correct += (pred_mood == tgt_mood).sum().item()

            # Energy MAE
            energy_pred = pred_chars[:, :, me:me+1].cpu().numpy()
            energy_tgt  = tgt_chars[:, :, me:me+1].cpu().numpy()
            energy_errs.append(np.abs(energy_pred - energy_tgt).mean())

            # Relationship MAE
            rel_errs.append(torch.abs(pred_rels - tgt_rels).mean().item())

            # Knowledge F1
            ge = me + 1 + len(GOALS)
            ke = ge + len(FACTS)
            pred_know = (torch.sigmoid(pred_chars[:, :, ge:ke]) > 0.5).float()
            tgt_know  = tgt_chars[:, :, ge:ke]
            know_tp += (pred_know * tgt_know).sum().item()
            know_fp += (pred_know * (1 - tgt_know)).sum().item()
            know_fn += ((1 - pred_know) * tgt_know).sum().item()

    know_precision = know_tp / (know_tp + know_fp + 1e-8)
    know_recall    = know_tp / (know_tp + know_fn + 1e-8)
    know_f1 = 2 * know_precision * know_recall / (know_precision + know_recall + 1e-8)

    return {
        "location_acc":    loc_correct / total,
        "mood_acc":        mood_correct / total,
        "energy_mae":      np.mean(energy_errs),
        "relationship_mae": np.mean(rel_errs),
        "knowledge_f1":    know_f1,
    }


def check_success(metrics: dict, phase0: bool = True) -> bool:
    """Check against success criteria.
    Phase 0 relaxes knowledge_f1 — training data has only 0.16% knowledge changes,
    so F1 is statistically meaningless. Knowledge is tested properly in Phase 1+.
    """
    criteria = {
        "location_acc":    (0.99, ">="),
        "mood_acc":        (0.95, ">="),
        "energy_mae":      (0.05, "<="),
        "relationship_mae":(0.05, "<="),
    }
    if not phase0:
        criteria["knowledge_f1"] = (0.90, ">=")
    print("\n── Success Criteria Check ──────────────────────────")
    all_pass = True
    for k, (threshold, op) in criteria.items():
        val = metrics.get(k, None)
        if val is None:
            continue
        if op == ">=":
            passed = val >= threshold
        else:
            passed = val <= threshold
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}  {k}: {val:.4f} (need {op} {threshold})")
    print("────────────────────────────────────────────────────")
    return all_pass


def plot_results(history: dict, metrics: dict, out_dir: Path, passed: bool):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    comp_keys = ["location", "mood", "energy", "relationships"]
    comp_data = {k: [c[k] for c in history["val_components"]] for k in comp_keys}

    BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"
    COMP_COLORS = [BLUE, ORANGE, GREEN, RED]

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f"Phase 0 — Baseline MLP  ·  {'ALL CRITERIA PASSED' if passed else 'FAILED'}",
        fontsize=15, fontweight="bold", color=GREEN if passed else RED, y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)

    # ── (A) Total loss curve ──
    ax_loss = fig.add_subplot(gs[0, :2])
    ax_loss.plot(epochs, history["train_loss"], label="Train", color=BLUE, lw=2)
    ax_loss.plot(epochs, history["val_loss"],   label="Val",   color=ORANGE, lw=2)
    ax_loss.set_title("(A) Total Loss — Train vs Val", fontsize=11)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)

    # ── (B) Per-component val losses ──
    ax_comp = fig.add_subplot(gs[0, 2])
    for k, c in zip(comp_keys, COMP_COLORS):
        ax_comp.plot(epochs, comp_data[k], label=k, color=c, lw=1.8)
    ax_comp.set_title("(B) Val Loss by Component", fontsize=11)
    ax_comp.set_xlabel("Epoch"); ax_comp.set_ylabel("Loss")
    ax_comp.legend(fontsize=8); ax_comp.grid(alpha=0.3)

    # ── (C) Accuracy / F1 bar chart ──
    ax_acc = fig.add_subplot(gs[1, :2])
    acc_items = [
        ("location_acc",  metrics["location_acc"],    0.99, ">="),
        ("mood_acc",      metrics["mood_acc"],         0.95, ">="),
        ("knowledge_f1",  metrics["knowledge_f1"],     0.90, ">="),
    ]
    names_a  = [x[0] for x in acc_items]
    values_a = [x[1] for x in acc_items]
    thresh_a = [x[2] for x in acc_items]
    ops_a    = [x[3] for x in acc_items]
    bcolors_a = [GREEN if (v >= t) else RED for v, t in zip(values_a, thresh_a)]

    bars_a = ax_acc.bar(names_a, values_a, color=bcolors_a, alpha=0.82, zorder=3)
    for bar, t in zip(bars_a, thresh_a):
        ax_acc.hlines(t, bar.get_x(), bar.get_x() + bar.get_width(),
                      colors="black", linestyles="--", lw=1.5, zorder=4)
    for bar in bars_a:
        ax_acc.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_acc.set_ylim(0.80, 1.05)
    ax_acc.set_title("(C) Accuracy & F1 (dashed = threshold)", fontsize=11)
    ax_acc.set_ylabel("Score"); ax_acc.grid(axis="y", alpha=0.3, zorder=0)

    # ── (D) MAE bar chart ──
    ax_err = fig.add_subplot(gs[1, 2])
    err_items = [
        ("energy\nMAE",       metrics["energy_mae"],       0.05, "<="),
        ("relationship\nMAE", metrics["relationship_mae"],  0.05, "<="),
    ]
    names_e  = [x[0] for x in err_items]
    values_e = [x[1] for x in err_items]
    thresh_e = [x[2] for x in err_items]
    bcolors_e = [GREEN if (v <= t) else RED for v, t in zip(values_e, thresh_e)]

    bars_e = ax_err.bar(names_e, values_e, color=bcolors_e, alpha=0.82, zorder=3)
    for bar, t in zip(bars_e, thresh_e):
        ax_err.hlines(t, bar.get_x(), bar.get_x() + bar.get_width(),
                      colors="black", linestyles="--", lw=1.5, zorder=4)
    for bar in bars_e:
        ax_err.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_err.set_ylim(0, 0.065)
    ax_err.set_title("(D) Error Metrics (dashed = threshold)", fontsize=11)
    ax_err.set_ylabel("MAE"); ax_err.grid(axis="y", alpha=0.3, zorder=0)

    out_path = out_dir / "phase0_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")
    return out_path


def main():
    device = "cpu"
    phase0 = True  # 2-char tiny world for Phase 0
    n_chars = 2 if phase0 else 5

    print("=" * 60)
    print("Phase 0: Minimal Pipeline Validation")
    print(f"Characters: {n_chars}, Device: {device}")
    print("=" * 60)

    # ── 1. Generate data ──
    print("\n[1/3] Generating episodes...")
    episodes = generate_episodes(
        num_episodes=600,
        episode_length=50,
        seed=42,
        phase0=phase0,
    )
    train_t, val_t, test_t = split_episodes(episodes, test_ratio=0.15, val_ratio=0.05)

    # ── 2. Train ──
    print("\n[2/3] Training baseline MLP...")
    out_dir = Path("experiments/outputs/phase0")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = train(
        transitions_train=train_t,
        transitions_val=val_t,
        n_chars=n_chars,
        epochs=80,
        batch_size=256,
        lr=1e-3,
        hidden_dim=256,
        device=device,
        save_path=str(out_dir / "best_model.pt"),
    )

    # ── 3. Evaluate ──
    print("\n[3/3] Evaluating on test set...")
    from torch.utils.data import DataLoader
    test_ds = TransitionDataset(test_t)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False,
                         collate_fn=collate_fn, num_workers=0)

    metrics = evaluate_accuracy(result["model"], test_dl, n_chars, device)
    print("\n── Test Metrics ─────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    passed = check_success(metrics, phase0=phase0)

    if passed:
        print("\nPhase 0 PASSED. Pipeline is valid. Proceed to Phase 1.")
    else:
        print("\nPhase 0 FAILED. Debug before proceeding.")

    # Save history
    import json
    history = result["history"]
    with open(out_dir / "history.json", "w") as f:
        json.dump({
            "train_loss":    [float(x) for x in history["train_loss"]],
            "val_loss":      [float(x) for x in history["val_loss"]],
            "val_components":[{k: float(v) for k, v in c.items()}
                              for c in history["val_components"]],
            "test_metrics":  {k: float(v) for k, v in metrics.items()},
        }, f, indent=2)

    # Plot
    plot_results(history, metrics, out_dir, passed)
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
