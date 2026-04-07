"""
Phase 3: GRU World Model (RSSM-lite)
T(z_t, a_t, h_{t-1}) → (z_{t+1}, h_t)

Pipeline:
    z_t   = VAE.encode(s_t)              # frozen
    x_t   = cat(z_t, action_emb_t)
    h_t   = GRU(x_t, h_{t-1})
    z'_t  = z_t + MLP(h_t)              # residual delta in latent space
    s'_t  = VAE.decode(z'_t)            # frozen

Advantage over Phase 2: temporal context h_t lets the model track
energy depletion trends, relationship accumulation, and mood trajectories
across multi-step rollouts.

Training: teacher-forced (ground-truth z_t fed at each step).
Evaluation: both 1-step (teacher-forced) and 5-step open-loop rollout.

Success criteria:
    1-step teacher-forced:
        location_acc  >= 99%
        mood_acc      >= 95%
        energy_mae    <= 0.05
        relationship_mae <= 0.05

    5-step open-loop:
        location_acc  >= 97%
        mood_acc      >= 90%
        energy_mae    <= 0.10
        relationship_mae <= 0.10
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import numpy as np

from core import encode_state, encode_action, flat_state_dim
from core.world import LOCATIONS, ACTION_TYPES
from core.tensorize import CHAR_DIM, rel_dim
from models.baseline_mlp import ActionEncoder, component_loss


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Sliding-window sequences of length seq_len from episode lists.
    Each sample: T consecutive (state, action, next_state) transitions.

    Pre-encodes all tensors at init so DataLoader workers don't touch Python objects.
    """

    def __init__(self, episodes: list[list[dict]], seq_len: int = 10):
        self.seq_len = seq_len
        self.seqs: list[dict] = []

        for ep in episodes:
            if len(ep) < seq_len:
                continue
            chars = ep[0]["characters"]
            for start in range(len(ep) - seq_len + 1):
                window = ep[start:start + seq_len]
                self.seqs.append(self._encode_window(window, chars))

    @staticmethod
    def _encode_window(window: list[dict], chars: list[str]) -> dict:
        state_list, nc_list, nr_list, ne_list = [], [], [], []
        at_list, ac_list, al_list, af_list, ag_list, ai_list = [], [], [], [], [], []

        for t in window:
            s_enc  = encode_state(t["state"],      chars)
            a_enc  = encode_action(t["action"],     chars)
            ns_enc = encode_state(t["next_state"],  chars)

            state_list.append(s_enc["flat"])
            nc_list.append(ns_enc["characters"])
            nr_list.append(ns_enc["relationships"])
            ne_list.append(ns_enc["environment"])

            at_list.append(a_enc["type"])
            ac_list.append(a_enc["target_char"])
            al_list.append(a_enc["target_loc"])
            af_list.append(a_enc["target_fact"])
            ag_list.append(a_enc["target_goal"])
            ai_list.append(torch.tensor(chars.index(t["actor"]), dtype=torch.long))

        return {
            "state_seq":   torch.stack(state_list),   # (T, state_dim)
            "next_chars":  torch.stack(nc_list),       # (T, n_chars, CHAR_DIM)
            "next_rels":   torch.stack(nr_list),       # (T, n_rels)
            "next_env":    torch.stack(ne_list),       # (T, env_dim)
            "action_type": torch.stack(at_list),       # (T,)
            "action_char": torch.stack(ac_list),       # (T,)
            "action_loc":  torch.stack(al_list),       # (T,)
            "action_fact": torch.stack(af_list),       # (T,)
            "action_goal": torch.stack(ag_list),       # (T,)
            "actor_idx":   torch.stack(ai_list),       # (T,)
        }

    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i]


def seq_collate_fn(batch):
    """Stack list of sequence dicts → batched tensors (B, T, ...)."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ─── Model ────────────────────────────────────────────────────────────────────

class GRUWorldModel(nn.Module):
    """
    Recurrent world model: T(z_t, a_t, h_{t-1}) → (z'_t, h_t).

    GRU input at each step: cat(z_t, action_emb, loc_oh, act_oh, actor_oh)
    GRU hidden state carries temporal context across steps.
    Head MLP: h_t → delta_z → z'_t = z_t + delta_z  (residual)
    """

    def __init__(self, z_dim: int, n_chars: int, hidden_dim: int = 256,
                 gru_hidden: int = 256):
        super().__init__()
        self.action_enc = ActionEncoder(n_chars)
        action_dim = self.action_enc.out_dim  # 28

        extra_dim   = len(LOCATIONS) + len(ACTION_TYPES) + n_chars  # 3+8+5=16
        input_dim   = z_dim + action_dim + extra_dim                 # 64+28+16=108

        self.gru = nn.GRU(input_dim, gru_hidden, batch_first=True)

        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gru_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

        self.z_dim      = z_dim
        self.gru_hidden = gru_hidden
        self.n_locs     = len(LOCATIONS)
        self.n_acts     = len(ACTION_TYPES)
        self.n_chars    = n_chars

    def forward(self, z_seq, atype_seq, achar_seq, aloc_seq, afact_seq,
                agoal_seq, actor_seq, h0=None):
        """
        z_seq:    (B, T, z_dim)
        *_seq:    (B, T)  long
        h0:       (1, B, gru_hidden) or None

        Returns:
            z_pred_seq (B, T, z_dim)
            h_T        (1, B, gru_hidden)
        """
        B, T, _ = z_seq.shape
        dev = z_seq.device

        def flat(x): return x.reshape(B * T)

        action_emb = self.action_enc(
            flat(atype_seq), flat(achar_seq), flat(aloc_seq),
            flat(afact_seq), flat(agoal_seq)
        ).view(B, T, -1)

        loc_oh = torch.zeros(B * T, self.n_locs, device=dev)
        loc_oh.scatter_(1, flat(aloc_seq).unsqueeze(1).clamp(0, self.n_locs - 1), 1.0)
        loc_oh = loc_oh.view(B, T, -1)

        act_oh = torch.zeros(B * T, self.n_acts, device=dev)
        act_oh.scatter_(1, flat(atype_seq).unsqueeze(1), 1.0)
        act_oh = act_oh.view(B, T, -1)

        actor_oh = torch.zeros(B * T, self.n_chars, device=dev)
        actor_oh.scatter_(1, flat(actor_seq).unsqueeze(1), 1.0)
        actor_oh = actor_oh.view(B, T, -1)

        x = torch.cat([z_seq, action_emb, loc_oh, act_oh, actor_oh], dim=-1)

        gru_out, h_T = self.gru(x, h0)      # (B, T, gru_hidden), (1, B, gru_hidden)
        delta_z = self.head(gru_out)         # (B, T, z_dim)

        return z_seq + delta_z, h_T          # residual

    def step(self, z, atype, achar, aloc, afact, agoal, actor, h=None):
        """
        Single GRU step: (B, z_dim) × action scalars → (B, z_dim), h.
        Used for scheduled sampling (step-by-step training) and open-loop rollout.
        """
        return self.forward(
            z.unsqueeze(1), atype.unsqueeze(1), achar.unsqueeze(1),
            aloc.unsqueeze(1), afact.unsqueeze(1), agoal.unsqueeze(1),
            actor.unsqueeze(1), h,
        )  # returns (B, 1, z_dim), h — caller squeezes dim 1


def _weighted_loss(losses: dict) -> torch.Tensor:
    """Same weighting as vae_recon_loss: energy×10, relationships×10."""
    return (losses["location"]
            + losses["mood"]
            + losses["energy"]       * 10.0
            + losses["goal"]
            + losses["knowledge"]
            + losses["relationships"] * 10.0
            + losses["time"]
            + losses["weather"]
            + losses["events"])


# ─── Trainer ──────────────────────────────────────────────────────────────────

def train(
    episodes_train: list[list[dict]],
    episodes_val:   list[list[dict]],
    vae,
    n_chars:    int,
    seq_len:    int  = 10,
    epochs:     int  = 60,
    batch_size: int  = 128,
    lr:         float = 1e-3,
    hidden_dim: int  = 256,
    gru_hidden: int  = 256,
    device:     str  = "cpu",
    save_path:  Optional[str] = None,
    ss_start:   float = 1.0,   # teacher-forcing ratio at epoch 0
    ss_end:     float = 0.5,   # teacher-forcing ratio at final epoch
) -> dict:
    """
    Scheduled sampling: teacher_forcing_ratio decays linearly from ss_start → ss_end.
    At each step, with probability (1 - tf_ratio) the model's own prediction is fed
    as the next input instead of the ground-truth z — bridging the train/eval gap.
    """

    z_dim = vae.z_dim

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    model = GRUWorldModel(z_dim, n_chars, hidden_dim, gru_hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    print(f"Building sequence datasets (seq_len={seq_len})...")
    train_ds = SequenceDataset(episodes_train, seq_len)
    val_ds   = SequenceDataset(episodes_val,   seq_len)
    print(f"  train: {len(train_ds):,} sequences  |  val: {len(val_ds):,} sequences")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=seq_collate_fn, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          collate_fn=seq_collate_fn, num_workers=0)

    history  = {"train_loss": [], "val_loss": [], "val_components": [],
                "tf_ratio": []}
    best_val = float("inf")

    for epoch in range(epochs):
        # Linearly decay teacher-forcing ratio
        tf_ratio = ss_start - (ss_start - ss_end) * epoch / max(epochs - 1, 1)

        # ── Train (step-by-step with scheduled sampling) ──
        model.train()
        train_losses = []

        for batch in train_dl:
            s_seq = batch["state_seq"].float().to(device)  # (B, T, D)
            B, T, D = s_seq.shape

            with torch.no_grad():
                z_gt_flat, _ = vae.encode(s_seq.view(B * T, D))
            z_gt = z_gt_flat.view(B, T, -1)  # ground-truth latents

            atype  = batch["action_type"].to(device)
            achar  = batch["action_char"].to(device)
            aloc   = batch["action_loc"].to(device)
            afact  = batch["action_fact"].to(device)
            agoal  = batch["action_goal"].to(device)
            actor  = batch["actor_idx"].to(device)

            all_preds = []
            h = None
            z_in = z_gt[:, 0]  # always start from ground-truth z_0

            for t in range(T):
                z_pred_t, h = model.step(
                    z_in, atype[:, t], achar[:, t], aloc[:, t],
                    afact[:, t], agoal[:, t], actor[:, t], h,
                )
                z_pred_t = z_pred_t.squeeze(1)  # (B, z_dim)
                all_preds.append(z_pred_t)

                # Scheduled sampling: next input is gt or prediction
                if t + 1 < T:
                    if np.random.random() < tf_ratio:
                        z_in = z_gt[:, t + 1]        # teacher force
                    else:
                        z_in = z_pred_t.detach()     # use own prediction

            z_pred_seq = torch.stack(all_preds, dim=1)  # (B, T, z_dim)

            pred_flat = vae.decode(z_pred_seq.reshape(B * T, -1))
            tgt = {
                "next_chars": batch["next_chars"].view(B * T, n_chars, CHAR_DIM).to(device),
                "next_rels":  batch["next_rels"].view(B * T, -1).to(device),
                "next_env":   batch["next_env"].view(B * T, -1).to(device),
            }
            losses = component_loss(pred_flat, tgt, n_chars)
            loss   = _weighted_loss(losses)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        sched.step()

        # ── Validate ──
        model.eval()
        val_losses = []
        val_comps  = {k: [] for k in ["location", "mood", "energy",
                                       "goal", "knowledge", "relationships"]}

        with torch.no_grad():
            for batch in val_dl:
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

                pred_flat = vae.decode(z_pred_seq.view(B * T, -1))
                tgt = {
                    "next_chars": batch["next_chars"].view(B * T, n_chars, CHAR_DIM).to(device),
                    "next_rels":  batch["next_rels"].view(B * T, -1).to(device),
                    "next_env":   batch["next_env"].view(B * T, -1).to(device),
                }
                losses = component_loss(pred_flat, tgt, n_chars)
                val_losses.append(_weighted_loss(losses).item())
                for k in val_comps:
                    val_comps[k].append(losses[k].item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        comp_means = {k: np.mean(v) for k, v in val_comps.items()}

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_components"].append(comp_means)
        history["tf_ratio"].append(tf_ratio)

        # Save on core metrics (same weighting as training loss, minus knowledge/env)
        core_val = (comp_means["location"]
                    + comp_means["mood"]
                    + comp_means["energy"]        * 10.0
                    + comp_means["relationships"] * 10.0)
        if core_val < best_val:
            best_val = core_val
            if save_path:
                torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} tf={tf_ratio:.2f} | "
                  f"train={train_loss:.4f} val={val_loss:.4f} | "
                  f"loc={comp_means['location']:.4f} mood={comp_means['mood']:.4f} "
                  f"energy={comp_means['energy']:.5f} rel={comp_means['relationships']:.5f}")

    print(f"\nBest core val: {best_val:.4f}")
    return {"model": model, "history": history}
