"""
MicroWorld: Deterministic Transition Engine + Separate Stochastic Event Layer.

KEY DESIGN: T(s, a) -> s' is PURE / DETERMINISTIC.
Stochastic events applied separately via apply_stochastic_events().
"""

from __future__ import annotations
from copy import deepcopy
from typing import Optional
import random

from .world import (
    WorldState, CharacterState, Action,
    MOODS, LOCATIONS, GOALS, FACTS, EVENTS
)

# ─── Mood helpers ─────────────────────────────────────────────────────────────

MOOD_VALENCE = {m: v for m, v in zip(MOODS, [-1, -2, -3, 2, -1])}
# calm=0, anxious=-1, angry=-2, happy=1, sad=-1 (rough ordering)
MOOD_ORDER = ["angry", "anxious", "sad", "calm", "happy"]

def _improve_mood(char: CharacterState) -> CharacterState:
    idx = MOOD_ORDER.index(char.mood)
    new_mood = MOOD_ORDER[min(idx + 1, len(MOOD_ORDER) - 1)]
    return CharacterState(location=char.location, mood=new_mood,
                          energy=char.energy, current_goal=char.current_goal,
                          knowledge=char.knowledge)

def _worsen_mood(char: CharacterState) -> CharacterState:
    idx = MOOD_ORDER.index(char.mood)
    new_mood = MOOD_ORDER[max(idx - 1, 0)]
    return CharacterState(location=char.location, mood=new_mood,
                          energy=char.energy, current_goal=char.current_goal,
                          knowledge=char.knowledge)

def _set_mood(char: CharacterState, mood: str) -> CharacterState:
    return CharacterState(location=char.location, mood=mood,
                          energy=char.energy, current_goal=char.current_goal,
                          knowledge=char.knowledge)

def _set_energy(char: CharacterState, energy: float) -> CharacterState:
    e = max(0.0, min(1.0, energy))
    return CharacterState(location=char.location, mood=char.mood,
                          energy=e, current_goal=char.current_goal,
                          knowledge=char.knowledge)

def _set_location(char: CharacterState, loc: str) -> CharacterState:
    return CharacterState(location=loc, mood=char.mood,
                          energy=char.energy, current_goal=char.current_goal,
                          knowledge=char.knowledge)

def _set_goal(char: CharacterState, goal: str) -> CharacterState:
    return CharacterState(location=char.location, mood=char.mood,
                          energy=char.energy, current_goal=goal,
                          knowledge=char.knowledge)

def _add_knowledge(char: CharacterState, fact: str) -> CharacterState:
    return CharacterState(location=char.location, mood=char.mood,
                          energy=char.energy, current_goal=char.current_goal,
                          knowledge=frozenset(char.knowledge | {fact}))

# ─── Location facts (observable public facts per location) ────────────────────

LOCATION_FACTS = {
    "寝宫":  frozenset(["秘密书信", "暗藏毒药", "私藏细软"]),
    "御花园": frozenset(["偷听密谈", "秘密结盟", "散布谣言"]),
    "凤仪宫": frozenset(["伪造懿旨", "盗取印信", "收买宫人"]),
}

# ─── Deterministic transition ─────────────────────────────────────────────────

def transition(state: WorldState, action: Action) -> WorldState:
    """
    Pure deterministic transition: T(state, action) -> next_state.
    No randomness. Same input always gives same output.
    """
    s = deepcopy(state)
    actor = action.actor
    char = s.characters[actor]

    if action.type == "move_to":
        assert action.target_loc in LOCATIONS
        s.characters[actor] = _set_energy(
            _set_location(char, action.target_loc),
            char.energy - 0.1
        )

    elif action.type == "talk_to":
        tgt = action.target_char
        if tgt not in s.characters:
            return state  # invalid target
        target = s.characters[tgt]
        if char.location != target.location:
            return state  # must be co-located
        rel = s.get_rel(actor, tgt)
        if rel > 0.5:
            s.characters[actor] = _improve_mood(char)
            s.characters[tgt] = _improve_mood(target)
        elif rel < -0.3:
            s.characters[actor] = _worsen_mood(char)
            s.characters[tgt] = _worsen_mood(target)
        # neutral: no mood change

    elif action.type == "argue":
        tgt = action.target_char
        if tgt not in s.characters:
            return state
        target = s.characters[tgt]
        # Relationship degrades for both directions
        s.relationships[(actor, tgt)] = max(-1.0, s.get_rel(actor, tgt) - 0.15)
        s.relationships[(tgt, actor)] = max(-1.0, s.get_rel(tgt, actor) - 0.10)
        s.characters[actor] = _set_energy(_set_mood(char, "angry"), char.energy - 0.2)
        s.characters[tgt] = _set_mood(target, "angry")

    elif action.type == "cooperate":
        tgt = action.target_char
        if tgt not in s.characters:
            return state
        if char.location != s.characters[tgt].location:
            return state  # must be co-located
        target = s.characters[tgt]
        s.relationships[(actor, tgt)] = min(1.0, s.get_rel(actor, tgt) + 0.10)
        s.relationships[(tgt, actor)] = min(1.0, s.get_rel(tgt, actor) + 0.08)
        s.characters[actor] = _set_mood(char, "happy")
        s.characters[tgt] = _improve_mood(target)

    elif action.type == "rest":
        s.characters[actor] = _set_energy(char, char.energy + 0.3)
        if char.mood in ("angry", "anxious"):
            s.characters[actor] = _improve_mood(s.characters[actor])

    elif action.type == "inform":
        tgt = action.target_char
        fact = action.target_fact
        if tgt not in s.characters or fact is None:
            return state
        target = s.characters[tgt]
        if char.location != target.location:
            return state
        # Deterministic: transfer always succeeds if co-located
        s.characters[tgt] = _add_knowledge(target, fact)

    elif action.type == "plan":
        goal = action.target_goal
        if goal in GOALS:
            s.characters[actor] = _set_goal(char, goal)

    elif action.type == "observe":
        loc_facts = LOCATION_FACTS.get(char.location, frozenset())
        updated = char
        for fact in loc_facts:
            updated = _add_knowledge(updated, fact)
        s.characters[actor] = updated

    elif action.type == "wait":
        pass  # only time progression

    elif action.type == "react":
        negative = {"嫔妃贬位", "宫中失火", "皇嗣夭折", "外敌入侵",
                    "宫人暴毙", "密报泄露", "皇上病危", "废后传言",
                    "逃宫风波", "赐死懿旨", "太后训诫", "阴谋败露"}
        positive = {"皇上临幸", "嫔妃晋封", "皇子降生", "宫宴举行",
                    "结盟成功", "真相大白", "圣恩浩荡", "贵人相助"}
        evt = action.target_event
        if evt in negative:
            s.characters[actor] = _worsen_mood(char)
        elif evt in positive:
            s.characters[actor] = _improve_mood(char)

    # ── Time progression (deterministic) ──
    s.tick += 1
    _update_time_of_day(s)
    _decay_energy(s)

    return s


def _update_time_of_day(s: WorldState) -> None:
    order = ["morning", "afternoon", "evening", "night"]
    # Every 6 ticks, advance time of day
    if s.tick % 6 == 0:
        idx = order.index(s.time_of_day)
        s.time_of_day = order[(idx + 1) % 4]


def _decay_energy(s: WorldState) -> None:
    for name, char in s.characters.items():
        s.characters[name] = _set_energy(char, char.energy - 0.01)


# ─── Stochastic event layer (SEPARATE from deterministic transition) ───────────

def apply_stochastic_events(state: WorldState, p_per_10: float = 0.3) -> WorldState:
    """
    Stochastic layer — called AFTER deterministic transition.
    Max 1 event per 10 ticks. Model is NOT trained to predict this.
    """
    if state.tick % 10 != 0:
        return state
    if random.random() > p_per_10:
        return state

    s = deepcopy(state)
    event = random.choice(EVENTS)
    new_events = frozenset(s.active_events | {event})
    s.active_events = new_events
    return s


# ─── Valid action generator ───────────────────────────────────────────────────

def valid_actions(state: WorldState, actor: str) -> list[Action]:
    """Return all valid actions for a given actor in the current state."""
    char = state.characters[actor]
    others = [n for n in state.characters if n != actor]
    actions = []

    # move_to: any location that is not current
    for loc in LOCATIONS:
        if loc != char.location:
            actions.append(Action(type="move_to", actor=actor, target_loc=loc))

    # talk_to, argue, cooperate: co-located characters
    co_located = [n for n in others
                  if state.characters[n].location == char.location]
    for tgt in co_located:
        actions.append(Action(type="talk_to", actor=actor, target_char=tgt))
        actions.append(Action(type="argue", actor=actor, target_char=tgt))
        actions.append(Action(type="cooperate", actor=actor, target_char=tgt))

    # inform: co-located characters, any fact actor knows
    for tgt in co_located:
        for fact in char.knowledge:
            actions.append(Action(type="inform", actor=actor,
                                  target_char=tgt, target_fact=fact))

    # rest, observe, wait: always valid
    actions.append(Action(type="rest", actor=actor))
    actions.append(Action(type="observe", actor=actor))
    actions.append(Action(type="wait", actor=actor))

    # plan: any goal different from current
    for goal in GOALS:
        if goal != char.current_goal:
            actions.append(Action(type="plan", actor=actor, target_goal=goal))

    # react: any active event
    for evt in state.active_events:
        actions.append(Action(type="react", actor=actor, target_event=evt))

    return actions
