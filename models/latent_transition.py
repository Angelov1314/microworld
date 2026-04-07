"""
Phase 2: Latent Transition Model
T(z, a) -> z'  in VAE latent space.

Pipeline:
    z  = VAE.encode(state)          # frozen
    z' = LatentTransitionModel(z, a) # trained here
    pred_state = VAE.decode(z')     # frozen

Advantages over baseline_mlp:
    - Predicts in 64-dim latent space (vs 242-dim observation space)
    - Smooth latent space from VAE helps transition learning
    - Residual: predict delta_z, not full z'
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from torch.utils.data import DataLoader

from core.world import LOCATIONS, ACTION_TYPES
from models.baseline_mlp import ActionEncoder, TransitionDataset, collate_fn, component_loss
from models.world_vae import WorldStateVAE


# ─── Model ────────────────────────────────────────────────────────────────────

class LatentTransitionModel(nn.Module):
    """
    T(z, a) -> z'  via residual delta prediction.
    Reuses ActionEncoder from baseline_mlp (type + char + loc + fact + goal embeddings).
    Also uses direct one-hots for target_loc, action_type, and actor — same as Phase 0.
    """
    def __init__(self, z_dim: int, n_chars: int, hidden_dim: int = 256):
        super().__init__()
        self.action_enc = ActionEncoder(n_chars)
        action_dim = self.action_enc.out_dim  # 28

        # Direct one-hot signals (no embedding compression)
        extra_dim = len(LOCATIONS) + len(ACTION_TYPES) + n_chars
        in_dim = z_dim + action_dim + extra_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.z_dim    = z_dim
        self.n_locs   = len(LOCATIONS)
        self.n_acts   = len(ACTION_TYPES)
        self.n_chars  = n_chars

    def forward(self, z, atype, achar, aloc, afact, agoal, actor_idx):
        action_emb = self.action_enc(atype, achar, aloc, afact, agoal)
        B = z.shape[0]

        loc_onehot = torch.zeros(B, self.n_locs, device=z.device)
        loc_onehot.scatter_(1, aloc.unsqueeze(1).clamp(0, self.n_locs - 1), 1.0)

        act_onehot = torch.zeros(B, self.n_acts, device=z.device)
        act_onehot.scatter_(1, atype.unsqueeze(1), 1.0)

        actor_onehot = torch.zeros(B, self.n_chars, device=z.device)
        actor_onehot.scatter_(1, actor_idx.unsqueeze(1), 1.0)

        x = torch.cat([z, action_emb, loc_onehot, act_onehot, actor_onehot], dim=-1)
        delta_z = self.net(x)
        return z + delta_z  # residual: predict change in latent space


# ─── Trainer ──────────────────────────────────────────────────────────────────

def train(
    transitions_train: list[dict],
    transitions_val: list[dict],
    vae: WorldStateVAE,
    n_chars: int,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> dict:

    z_dim = vae.z_dim

    # Freeze VAE — we only train the transition model
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    model = LatentTransitionModel(z_dim, n_chars, hidden_dim).to(device)
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

            # Encode state with frozen VAE (no gradient needed for encoder)
            with torch.no_grad():
                z, _ = vae.encode(s)  # use mu (deterministic)

            # Predict next latent
            z_pred = model(z, at, ac, al, af, ag, ai)

            # Decode prediction (gradients flow through decoder to z_pred)
            pred_flat = vae.decode(z_pred)

            losses = component_loss(
                pred_flat,
                {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)},
                n_chars,
            )
            # Upweight continuous features to match CE scale (same as vae_recon_loss)
            loss = (losses["location"]
                    + losses["mood"]
                    + losses["energy"] * 10.0
                    + losses["goal"]
                    + losses["knowledge"]
                    + losses["relationships"] * 10.0
                    + losses["time"]
                    + losses["weather"]
                    + losses["events"])

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

                z, _ = vae.encode(s)
                z_pred = model(z, at, ac, al, af, ag, ai)
                pred_flat = vae.decode(z_pred)

                losses = component_loss(
                    pred_flat,
                    {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)},
                    n_chars,
                )
                val_loss_weighted = (losses["location"]
                                     + losses["mood"]
                                     + losses["energy"] * 10.0
                                     + losses["goal"]
                                     + losses["knowledge"]
                                     + losses["relationships"] * 10.0
                                     + losses["time"]
                                     + losses["weather"]
                                     + losses["events"])
                val_losses.append(val_loss_weighted.item())
                for k in val_comps:
                    val_comps[k].append(losses[k].item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        comp_means = {k: np.mean(v) for k, v in val_comps.items()}

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_components"].append(comp_means)

        # Save based on core metrics only (same weighting as vae_recon_loss)
        core_val = (comp_means["location"]
                    + comp_means["mood"]
                    + comp_means["energy"] * 10.0
                    + comp_means["relationships"] * 10.0)
        if core_val < best_val:
            best_val = core_val
            if save_path:
                torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train={train_loss:.4f} val={val_loss:.4f} | "
                  f"loc={comp_means['location']:.4f} mood={comp_means['mood']:.4f} "
                  f"energy={comp_means['energy']:.5f} rel={comp_means['relationships']:.5f}")

    print(f"\nBest core val loss (loc+mood+energy×10+rel×10): {best_val:.4f}")
    return {"model": model, "history": history}
