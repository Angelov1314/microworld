"""
Phase 1: WorldStateVAE
Encodes flat world state → latent z (μ, log_var) → reconstructed state.
Loss: per-component reconstruction + β*KL divergence (β-warmup).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import numpy as np

from core import encode_state, flat_state_dim
from models.baseline_mlp import component_loss


# ─── Dataset ──────────────────────────────────────────────────────────────────

class StateDataset(Dataset):
    """
    States from transitions (both s and s') for self-supervised VAE training.

    Why both s and s'?  2× more data, no cross-fold leakage (split is by episode,
    so both states from the same transition always land in the same fold).

    Key naming: `next_*` keys are used to match the component_loss() API from
    baseline_mlp.py — even though half the samples are current states, not next states.
    The loss function is symmetric: it only cares that input == target for reconstruction.
    """
    def __init__(self, transitions: list[dict]):
        self.data = []
        for t in transitions:
            chars = t["characters"]
            for sk in ("state", "next_state"):
                enc = encode_state(t[sk], chars)
                self.data.append({
                    "state_flat": enc["flat"],           # encoder input AND recon target
                    "next_chars": enc["characters"],     # recon target (component_loss API)
                    "next_rels":  enc["relationships"],  # recon target (component_loss API)
                    "next_env":   enc["environment"],    # recon target (component_loss API)
                    "char_names": chars,
                })

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def state_collate(batch):
    keys = [k for k in batch[0] if k != "char_names"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["char_names"] = batch[0]["char_names"]
    return out


# ─── Model ────────────────────────────────────────────────────────────────────

class WorldStateVAE(nn.Module):
    """
    VAE: flat_state → (μ, log_var) → z → reconstructed flat_state.
    Encoder and decoder are symmetric MLPs.
    At eval time, reparameterize uses μ directly (deterministic).
    """
    def __init__(self, state_dim: int, z_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu  # deterministic at eval

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL(N(μ,σ²) ‖ N(0,1)), averaged over batch and latent dims."""
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()


def vae_recon_loss(component_losses: dict) -> torch.Tensor:
    """
    Weighted reconstruction loss for VAE.

    Problem: categorical CE losses (location ~1.0, mood ~0.5) are 10-50× larger
    than continuous MSE losses (energy ~0.005, relationships ~0.06). Without
    upweighting, the VAE bottleneck discards continuous features entirely.

    Fix: upweight energy and relationships so they compete with categoricals.
    """
    return (
        component_losses["location"]              # 3× CE (already upweighted)
        + component_losses["mood"]                # 1× CE
        + component_losses["energy"] * 10.0       # MSE: upweight to match CE scale
        + component_losses["goal"]                # 1× CE
        + component_losses["knowledge"] * 0.5     # BCE with pos_weight=10; scale down to not dominate
        + component_losses["relationships"] * 10.0  # MSE: upweight to match CE scale
        + component_losses["time"]                # CE
        + component_losses["weather"]             # CE
        + component_losses["events"]              # BCE
    )


# ─── Trainer ──────────────────────────────────────────────────────────────────

def train(
    transitions_train: list[dict],
    transitions_val: list[dict],
    n_chars: int,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 3e-4,
    hidden_dim: int = 256,
    z_dim: int = 64,
    beta_max: float = 1.0,
    beta_warmup_epochs: int = 20,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> dict:

    state_dim = flat_state_dim(n_chars)
    model = WorldStateVAE(state_dim, z_dim, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    train_ds = StateDataset(transitions_train)
    val_ds   = StateDataset(transitions_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=state_collate, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          collate_fn=state_collate, num_workers=0)

    history = {
        "train_loss": [], "val_loss": [],
        "val_kl": [], "val_recon": [],
        "val_components": [],
    }
    best_val = float("inf")

    for epoch in range(epochs):
        beta = min(beta_max, beta_max * epoch / max(beta_warmup_epochs, 1))

        # ── Train ──
        model.train()
        train_losses = []
        for batch in train_dl:
            x = batch["state_flat"].float().to(device)
            recon, mu, log_var = model(x)
            recon_losses = component_loss(
                recon,
                {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)},
                n_chars,
            )
            loss = vae_recon_loss(recon_losses) + beta * kl_loss(mu, log_var)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_losses, val_kls, val_recons = [], [], []
        val_comps = {k: [] for k in ["location", "mood", "energy", "relationships"]}
        with torch.no_grad():
            for batch in val_dl:
                x = batch["state_flat"].float().to(device)
                recon, mu, log_var = model(x)
                recon_losses = component_loss(
                    recon,
                    {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)},
                    n_chars,
                )
                kl = kl_loss(mu, log_var)
                recon_loss_val = vae_recon_loss(recon_losses)
                val_losses.append((recon_loss_val + beta * kl).item())
                val_kls.append(kl.item())
                val_recons.append(recon_loss_val.item())
                for k in val_comps:
                    val_comps[k].append(recon_losses[k].item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        comp_means = {k: np.mean(v) for k, v in val_comps.items()}

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_kl"].append(float(np.mean(val_kls)))
        history["val_recon"].append(float(np.mean(val_recons)))
        history["val_components"].append(comp_means)

        if val_loss < best_val:
            best_val = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} β={beta:.2f} | "
                  f"train={train_loss:.4f} val={val_loss:.4f} | "
                  f"recon={np.mean(val_recons):.4f} kl={np.mean(val_kls):.4f} | "
                  f"loc={comp_means['location']:.4f} mood={comp_means['mood']:.4f}")

    print(f"\nBest val loss: {best_val:.4f}")
    return {"model": model, "history": history}
