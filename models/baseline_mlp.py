"""
Phase 0 / Ablation Baseline: Direct MLP transition model.
flat_state + action_onehot -> flat_next_state

No autoencoder. No latent space. Just a raw MLP.
This is the ablation baseline to beat with the VAE-based model.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import numpy as np

from core import encode_state, encode_action, flat_state_dim, ACTION_TYPES


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TransitionDataset(Dataset):
    def __init__(self, transitions: list[dict]):
        self.data = []
        for t in transitions:
            characters = t["characters"]
            s_enc  = encode_state(t["state"],      characters)
            a_enc  = encode_action(t["action"],     characters)
            ns_enc = encode_state(t["next_state"],  characters)

            actor_idx = torch.tensor(characters.index(t["actor"]), dtype=torch.long)
            self.data.append({
                "state_flat":      s_enc["flat"],
                "action_type":     a_enc["type"],
                "action_char":     a_enc["target_char"],
                "action_loc":      a_enc["target_loc"],
                "action_fact":     a_enc["target_fact"],
                "action_goal":     a_enc["target_goal"],
                "actor_idx":       actor_idx,
                "next_state_flat": ns_enc["flat"],
                # Per-component tensors for detailed evaluation
                "next_chars":      ns_enc["characters"],       # (n, 39)
                "next_rels":       ns_enc["relationships"],    # (n*(n-1),)
                "next_env":        ns_enc["environment"],      # (27,)
                "characters":      characters,
            })

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate_fn(batch):
    """Custom collate to handle list fields."""
    keys = [k for k in batch[0] if k != "characters"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["characters"] = batch[0]["characters"]  # same for all (fixed world)
    return out


# ─── Model ────────────────────────────────────────────────────────────────────

class ActionEncoder(nn.Module):
    """Encode parameterized action tuple into a fixed vector."""
    def __init__(self, n_chars: int, out_dim: int = 24):
        super().__init__()
        self.type_emb = nn.Embedding(len(ACTION_TYPES), 8)
        self.char_emb = nn.Embedding(n_chars + 1, 6)     # +1 for null
        self.loc_emb  = nn.Embedding(4, 4)                # 3 locs + null
        self.fact_emb = nn.Embedding(21, 6)               # 20 facts + null
        self.goal_emb = nn.Embedding(11, 4)               # 10 goals + null
        self.out_dim = 8 + 6 + 4 + 6 + 4  # = 28

    def forward(self, atype, achar, aloc, afact, agoal):
        return torch.cat([
            self.type_emb(atype),
            self.char_emb(achar),
            self.loc_emb(aloc),
            self.fact_emb(afact),
            self.goal_emb(agoal),
        ], dim=-1)


class BaselineMLP(nn.Module):
    """
    Direct transition model: flat_state + action -> flat_next_state.
    Residual: predicts delta (change), not full next state.

    Key: target_loc is ALSO one-hot concatenated directly alongside embeddings
    to give a strong unambiguous signal for move_to location prediction.
    """
    def __init__(self, state_dim: int, n_chars: int, hidden_dim: int = 256):
        super().__init__()
        self.action_enc = ActionEncoder(n_chars)
        action_dim = self.action_enc.out_dim
        # +len(LOCATIONS) for direct one-hot target_loc signal
        # +len(ACTION_TYPES) for direct one-hot action_type signal
        # +n_chars for actor one-hot (tells model which char slot is acting)
        extra_dim = len(LOCATIONS) + len(ACTION_TYPES) + n_chars
        in_dim = state_dim + action_dim + extra_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.n_locs = len(LOCATIONS)
        self.n_acts = len(ACTION_TYPES)
        self.n_chars = n_chars

    def forward(self, state_flat, atype, achar, aloc, afact, agoal, actor_idx):
        action_emb = self.action_enc(atype, achar, aloc, afact, agoal)

        # Direct one-hot signals (no embedding compression)
        B = state_flat.shape[0]
        loc_onehot = torch.zeros(B, self.n_locs, device=state_flat.device)
        loc_onehot.scatter_(1, aloc.unsqueeze(1).clamp(0, self.n_locs - 1), 1.0)

        act_onehot = torch.zeros(B, self.n_acts, device=state_flat.device)
        act_onehot.scatter_(1, atype.unsqueeze(1), 1.0)

        # Actor one-hot: tells model which character slot is performing the action
        actor_onehot = torch.zeros(B, self.n_chars, device=state_flat.device)
        actor_onehot.scatter_(1, actor_idx.unsqueeze(1), 1.0)

        x = torch.cat([state_flat, action_emb, loc_onehot, act_onehot, actor_onehot], dim=-1)
        delta = self.net(x)
        return state_flat + delta  # residual: predict change


# ─── Loss ─────────────────────────────────────────────────────────────────────

from core.tensorize import (
    CHAR_DIM, ENV_DIM,
    LOCATIONS, MOODS, GOALS, FACTS, EVENTS,
)

def component_loss(pred_flat: torch.Tensor, target_batch: dict,
                   n_chars: int) -> dict:
    """
    Decompose flat prediction tensor into per-component losses.
    Returns dict of named losses + total.
    """
    from core.tensorize import rel_dim
    n_rels = rel_dim(n_chars)

    # Parse predicted flat tensor back into components
    char_end = n_chars * CHAR_DIM
    rel_end  = char_end + n_rels

    pred_chars = pred_flat[:, :char_end].view(-1, n_chars, CHAR_DIM)  # (B, n, 39)
    pred_rels  = pred_flat[:, char_end:rel_end]                         # (B, n_rels)
    pred_env   = pred_flat[:, rel_end:]                                  # (B, 27)

    tgt_chars  = target_batch["next_chars"].float()   # (B, n, 39)
    tgt_rels   = target_batch["next_rels"].float()    # (B, n_rels)
    tgt_env    = target_batch["next_env"].float()     # (B, 27)

    losses = {}

    # ── Per character ──
    loc_start  = 0
    loc_end    = len(LOCATIONS)
    mood_end   = loc_end + len(MOODS)
    energy_end = mood_end + 1
    goal_end   = energy_end + len(GOALS)
    know_end   = goal_end + len(FACTS)

    # Location: cross-entropy on logits (upweighted — this is key for narrative)
    pred_loc = pred_chars[:, :, loc_start:loc_end]   # (B, n, 3)
    tgt_loc  = tgt_chars[:, :, loc_start:loc_end].argmax(dim=-1)  # (B, n)
    losses["location"] = 3.0 * F.cross_entropy(
        pred_loc.reshape(-1, len(LOCATIONS)), tgt_loc.reshape(-1))

    # Mood: cross-entropy
    pred_mood = pred_chars[:, :, loc_end:mood_end]
    tgt_mood  = tgt_chars[:, :, loc_end:mood_end].argmax(dim=-1)
    losses["mood"] = F.cross_entropy(
        pred_mood.reshape(-1, len(MOODS)), tgt_mood.reshape(-1))

    # Energy: MSE
    pred_energy = pred_chars[:, :, mood_end:energy_end]
    tgt_energy  = tgt_chars[:, :, mood_end:energy_end]
    losses["energy"] = F.mse_loss(pred_energy, tgt_energy)

    # Goal: cross-entropy
    pred_goal = pred_chars[:, :, energy_end:goal_end]
    tgt_goal  = tgt_chars[:, :, energy_end:goal_end].argmax(dim=-1)
    losses["goal"] = F.cross_entropy(
        pred_goal.reshape(-1, len(GOALS)), tgt_goal.reshape(-1))

    # Knowledge: binary cross-entropy (multi-hot)
    pred_know = pred_chars[:, :, goal_end:know_end]
    tgt_know  = tgt_chars[:, :, goal_end:know_end]
    losses["knowledge"] = F.binary_cross_entropy_with_logits(pred_know, tgt_know)

    # ── Relationships: MSE ──
    losses["relationships"] = F.mse_loss(pred_rels, tgt_rels)

    # ── Environment ──
    # time_of_day: one-hot 4
    losses["time"] = F.cross_entropy(
        pred_env[:, :4],
        tgt_env[:, :4].argmax(dim=-1))
    # weather: one-hot 3
    losses["weather"] = F.cross_entropy(
        pred_env[:, 4:7],
        tgt_env[:, 4:7].argmax(dim=-1))
    # active_events: multi-hot 20
    losses["events"] = F.binary_cross_entropy_with_logits(
        pred_env[:, 7:], tgt_env[:, 7:])

    # Total
    losses["total"] = sum(losses.values())
    return losses


# ─── Trainer ──────────────────────────────────────────────────────────────────

def train(
    transitions_train: list[dict],
    transitions_val: list[dict],
    n_chars: int,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    hidden_dim: int = 256,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> dict:

    state_dim = flat_state_dim(n_chars)
    model = BaselineMLP(state_dim, n_chars, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    train_ds = TransitionDataset(transitions_train)
    val_ds   = TransitionDataset(transitions_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

    history = {"train_loss": [], "val_loss": [], "val_components": []}
    best_val = float("inf")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_losses = []
        for batch in train_dl:
            s   = batch["state_flat"].float().to(device)
            at  = batch["action_type"].to(device)
            ac  = batch["action_char"].to(device)
            al  = batch["action_loc"].to(device)
            af  = batch["action_fact"].to(device)
            ag  = batch["action_goal"].to(device)
            ai  = batch["actor_idx"].to(device)

            pred = model(s, at, ac, al, af, ag, ai)
            losses = component_loss(pred, {k: v.to(device) for k, v in batch.items()
                                           if isinstance(v, torch.Tensor)}, n_chars)
            loss = losses["total"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_losses = []
        val_comps  = {k: [] for k in ["location", "mood", "energy",
                                       "goal", "knowledge", "relationships"]}
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
                losses = component_loss(pred, {k: v.to(device) for k, v in batch.items()
                                               if isinstance(v, torch.Tensor)}, n_chars)
                val_losses.append(losses["total"].item())
                for k in val_comps:
                    val_comps[k].append(losses[k].item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        comp_means = {k: np.mean(v) for k, v in val_comps.items()}

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_components"].append(comp_means)

        if val_loss < best_val:
            best_val = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | "
                  f"loc={comp_means['location']:.4f} mood={comp_means['mood']:.4f} "
                  f"energy={comp_means['energy']:.5f} rel={comp_means['relationships']:.5f}")

    print(f"\nBest val loss: {best_val:.4f}")
    return {"model": model, "history": history}
