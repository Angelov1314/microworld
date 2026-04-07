"""
Phase 5: Cross-Entropy Method (CEM) Planner

Upgrade from random shooting:
  - Random shooting: sample N sequences uniformly, pick the best
  - CEM: iteratively refine a distribution over action sequences
    1. Sample N sequences from current distribution (factored categorical)
    2. Keep top-K elite sequences by cumulative reward
    3. Refit the distribution to the elites (MLE update)
    4. Repeat for n_iters iterations
    5. Return the first action of the best sequence found

CEM optimizes over the FULL joint (action_type, parameter) space — not just
action types — so it can identify both the best action and the best target
character/location. This is critical because the reward depends heavily on
which character is targeted (e.g., cooperate with 皇上 builds imperial favor).

The enumerated action space has ~41 discrete options per step.
Entropy floor prevents distribution collapse (model over-exploitation).
"""

from __future__ import annotations
import random
import torch
import numpy as np
from typing import Optional

from core.world import (
    Action, LOCATIONS, MOODS, GOALS, FACTS, ACTION_TYPES, CHARACTERS_FULL,
)
from core.tensorize import encode_action
from models.planner import (
    compute_reward, encode_action_batch, MOOD_SCORES, sample_random_action,
)


# ─── CEM Planner ──────────────────────────────────────────────────────────────

class CEMPlanner:
    """
    Cross-Entropy Method planner for the GRU world model.

    Optimizes over a discrete enumeration of all (type, parameter) action pairs,
    so CEM refines BOTH which action type to use AND which character/location to
    target. This is strictly more expressive than type-only CEM.

    At each planning call:
      - Maintains per-step categorical distribution over ALL_ACTIONS (~41 classes)
      - Runs n_iters CEM iterations:
          1. Sample N action sequences from current joint distribution
          2. Evaluate via GRU rollout
          3. Update distribution from top-K elite sequences
          4. Apply entropy floor to prevent model-exploitation collapse
      - Returns best action found

    Args:
        gru_model:   GRUWorldModel (frozen)
        vae:         WorldStateVAE (frozen)
        n_chars:     number of characters
        n_samples:   sequences sampled per CEM iteration
        horizon:     rollout steps
        n_iters:     CEM iterations per planning call
        elite_frac:  fraction of samples kept as elites (e.g. 0.1 = top 10%)
        gamma:       discount factor
        alpha:       smoothing coefficient for distribution update (momentum)
        n_inform_facts: number of facts to include for inform actions (limits space)
        actor:       character being planned for
        characters:  ordered character list
        device:      torch device
    """

    def __init__(
        self,
        gru_model,
        vae,
        n_chars:         int,
        n_samples:       int   = 200,
        horizon:         int   = 5,
        n_iters:         int   = 4,
        elite_frac:      float = 0.1,
        gamma:           float = 0.95,
        alpha:           float = 0.7,
        n_inform_facts:  int   = 3,
        actor:           str   = "甄嬛",
        characters:      Optional[list[str]] = None,
        device:          str   = "cpu",
        seed:            int   = 0,
    ):
        self.model          = gru_model
        self.vae            = vae
        self.n_chars        = n_chars
        self.n_samples      = n_samples
        self.horizon        = horizon
        self.n_iters        = n_iters
        self.n_elites       = max(1, int(n_samples * elite_frac))
        self.gamma          = gamma
        self.alpha          = alpha
        self.actor          = actor
        self.characters     = characters or CHARACTERS_FULL
        self.device         = device
        self.rng            = random.Random(seed)

        # Build the joint action enumeration
        self.all_actions = self._build_action_space(n_inform_facts)
        self.n_actions   = len(self.all_actions)

        self.model.eval()
        self.vae.eval()
        for p in self.model.parameters():  p.requires_grad = False
        for p in self.vae.parameters():    p.requires_grad = False

    def _build_action_space(self, n_inform_facts: int) -> list[Action]:
        """
        Enumerate all discrete (type, parameter) actions for self.actor.
        Limits inform actions to the top n_inform_facts facts to keep space tractable.
        """
        others  = [c for c in self.characters if c != self.actor]
        facts   = FACTS[:n_inform_facts]
        actions = []

        for atype in ACTION_TYPES:
            if atype == "move_to":
                for loc in LOCATIONS:
                    actions.append(Action(type=atype, actor=self.actor,
                                          target_loc=loc))
            elif atype in ("talk_to", "argue", "cooperate"):
                for ch in others:
                    actions.append(Action(type=atype, actor=self.actor,
                                          target_char=ch))
            elif atype == "inform":
                for ch in others:
                    for fact in facts:
                        actions.append(Action(type=atype, actor=self.actor,
                                              target_char=ch, target_fact=fact))
            elif atype == "plan":
                for goal in GOALS:
                    actions.append(Action(type=atype, actor=self.actor,
                                          target_goal=goal))
            else:  # rest, observe, wait, react
                actions.append(Action(type=atype, actor=self.actor))

        return actions

    @torch.no_grad()
    def _evaluate_sequences(
        self,
        seqs:    list[list[Action]],
        z_start: torch.Tensor,
        h_start: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out N action sequences and return (cumulative_reward, h_after_step1).
        cumulative_reward: (N,)
        h_after_step1:     (1, N, gru_hidden)
        """
        N   = len(seqs)
        dev = self.device

        z_batch = z_start.unsqueeze(0).expand(N, -1).clone()
        h_batch = h_start.expand(1, N, -1).clone() if h_start is not None else None

        cumulative_reward = torch.zeros(N, device=dev)
        h_after_step1     = None

        for step in range(self.horizon):
            actions_t = [seqs[i][step] for i in range(N)]
            enc = encode_action_batch(actions_t, self.characters, self.actor, dev)

            z_pred, h_batch = self.model.step(
                z_batch,
                enc["action_type"], enc["action_char"], enc["action_loc"],
                enc["action_fact"], enc["action_goal"], enc["actor_idx"],
                h_batch,
            )
            z_batch = z_pred.squeeze(1)

            if step == 0:
                h_after_step1 = h_batch

            pred_flat = self.vae.decode(z_batch)
            r = compute_reward(pred_flat, self.n_chars)
            cumulative_reward += (self.gamma ** step) * r

        return cumulative_reward, h_after_step1

    @torch.no_grad()
    def plan(
        self,
        z:  torch.Tensor,
        h:  Optional[torch.Tensor],
    ) -> tuple[Action, torch.Tensor]:
        """
        CEM planning: iteratively refine joint (type, parameter) action distribution.

        Returns:
            best_action: Action to execute
            h_next:      (1, 1, gru_hidden) hidden state after first step
        """
        # Uniform initial distribution over ALL actions at each horizon step
        # logits shape: (horizon, n_actions)
        logits = torch.zeros(self.horizon, self.n_actions, device=self.device)

        best_action = None
        best_reward = -float("inf")
        best_h_next = None

        for iteration in range(self.n_iters):
            # Sample actions from current joint distribution
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (H, n_actions)

            # Use self.rng (seeded Python random) — not numpy global state.
            # This ensures deterministic, reproducible sampling isolated from
            # numpy operations inside model inference.
            # Store (action_idx, action) pairs to avoid any reverse-lookup cost.
            seqs_idx = []   # list of list of int (action indices)
            seqs_act = []   # list of list of Action
            for _ in range(self.n_samples):
                seq_idx, seq_act = [], []
                for t in range(self.horizon):
                    act_idx = self.rng.choices(
                        range(self.n_actions),
                        weights=probs[t].tolist(),
                        k=1,
                    )[0]
                    seq_idx.append(act_idx)
                    seq_act.append(self.all_actions[act_idx])
                seqs_idx.append(seq_idx)
                seqs_act.append(seq_act)

            # Evaluate using action objects
            rewards, h_after1 = self._evaluate_sequences(seqs_act, z, h)

            # Track global best
            top_reward, top_idx = rewards.max(0)
            if top_reward.item() > best_reward:
                best_reward = top_reward.item()
                best_action = seqs_act[top_idx.item()][0]
                best_h_next = h_after1[:, top_idx.item():top_idx.item()+1, :].contiguous()

            # Elite update: refit distribution from top-K sequences
            _, elite_idxs = torch.topk(rewards, self.n_elites)

            # Count action-index frequencies per step among elites (O(1) via stored idx)
            elite_counts = torch.zeros_like(logits)
            for i in elite_idxs.tolist():
                for t, act_idx in enumerate(seqs_idx[i]):
                    elite_counts[t, act_idx] += 1.0
            new_probs = elite_counts / self.n_elites  # (H, n_actions), sums to 1

            # Entropy floor: prevent collapse to a single (type, param) pair.
            # Without this, CEM over-exploits model artifacts and picks specific
            # (character, action) combos that look good in the model but not reality.
            # 8% floor gives enough exploration to be robust without fully killing
            # the distribution refinement signal.
            uniform   = torch.ones_like(new_probs) / self.n_actions
            new_probs = 0.92 * new_probs + 0.08 * uniform

            # Blend in probability space (standard CEM), convert back to logits
            old_probs = torch.softmax(logits, dim=-1)
            blended   = self.alpha * old_probs + (1.0 - self.alpha) * new_probs
            logits    = torch.log(blended + 1e-8)

        return best_action, best_h_next
