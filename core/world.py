"""
MicroWorld: Narrative World State Definition
Phase 0: 2 characters, 1 location, 5 actions, no knowledge, no stochastic events.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Optional
import random

# ─── Enums / Constants ────────────────────────────────────────────────────────

LOCATIONS = ["寝宫", "御花园", "凤仪宫"]
MOODS = ["calm", "anxious", "angry", "happy", "sad"]
GOALS = [
    "争宠",     # compete for imperial favor
    "复仇",     # seek revenge
    "保全性命", # self-preservation
    "结盟",     # form alliances
    "揭露阴谋", # expose schemes
    "保护子嗣", # protect heirs
    "获得权势", # gain power
    "寻找真情", # seek genuine affection
    "明哲保身", # lay low and survive
    "争夺嫡位", # compete for status
]
FACTS = [
    "皇上驾临", "秘密书信", "暗藏毒药", "嫁祸阴谋", "宫中密道",
    "假怀龙嗣", "私藏细软", "勾结外臣", "私制禁药", "偷听密谈",
    "伪造懿旨", "通敌叛国", "秘密结盟", "替人受过", "暗中监视",
    "盗取印信", "散布谣言", "收买宫人", "藏匿证据", "揭发告密",
]
EVENTS = [
    # positive
    "皇上临幸", "嫔妃晋封", "皇子降生", "宫宴举行",
    "结盟成功", "真相大白", "圣恩浩荡", "贵人相助",
    # negative
    "嫔妃贬位", "宫中失火", "皇嗣夭折", "外敌入侵",
    "宫人暴毙", "密报泄露", "皇上病危", "废后传言",
    "逃宫风波", "赐死懿旨", "太后训诫", "阴谋败露",
]

CHARACTERS_PHASE0 = ["甄嬛", "皇后"]                              # Phase 0: 2 characters
CHARACTERS_FULL   = ["甄嬛", "皇后", "安陵容", "皇上", "沈眉庄"]  # Phase 1+: 5 characters

ACTION_TYPES = ["move_to", "talk_to", "argue", "cooperate", "rest",
                "inform", "plan", "observe", "wait", "react"]

# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class CharacterState:
    location: str = "寝宫"
    mood: str = "calm"
    energy: float = 1.0
    current_goal: str = "earn_money"
    knowledge: frozenset = field(default_factory=frozenset)  # immutable set for hashing

    def __post_init__(self):
        assert self.location in LOCATIONS, f"Invalid location: {self.location}"
        assert self.mood in MOODS, f"Invalid mood: {self.mood}"
        assert 0.0 <= self.energy <= 1.0, f"Energy out of range: {self.energy}"
        assert self.current_goal in GOALS, f"Invalid goal: {self.current_goal}"

    def with_knowledge(self, fact: str) -> "CharacterState":
        """Return new CharacterState with added fact (immutable update)."""
        return CharacterState(
            location=self.location,
            mood=self.mood,
            energy=self.energy,
            current_goal=self.current_goal,
            knowledge=frozenset(self.knowledge | {fact})
        )


@dataclass
class WorldState:
    tick: int = 0
    time_of_day: str = "morning"   # morning | afternoon | evening | night
    weather: str = "sunny"          # sunny | rainy | cloudy
    characters: dict = field(default_factory=dict)   # name -> CharacterState
    active_events: frozenset = field(default_factory=frozenset)
    # Directed relationship matrix: (from, to) -> float in [-1, 1]
    relationships: dict = field(default_factory=dict)

    def get_rel(self, a: str, b: str) -> float:
        return self.relationships.get((a, b), 0.0)

    def set_rel(self, a: str, b: str, val: float) -> "WorldState":
        """Return new WorldState with updated relationship (clamped)."""
        new = deepcopy(self)
        new.relationships[(a, b)] = max(-1.0, min(1.0, val))
        return new


@dataclass
class Action:
    type: str
    actor: str
    target_char: Optional[str] = None    # for talk_to, argue, cooperate, inform
    target_loc: Optional[str] = None     # for move_to
    target_fact: Optional[str] = None   # for inform
    target_goal: Optional[str] = None   # for plan
    target_event: Optional[str] = None  # for react


# ─── World Initialization ─────────────────────────────────────────────────────

def make_phase0_world() -> WorldState:
    """Phase 0: 甄嬛 vs 皇后, minimal state."""
    chars = {
        "甄嬛": CharacterState(location="寝宫", mood="calm",  energy=0.8, current_goal="争宠"),
        "皇后": CharacterState(location="寝宫", mood="calm",  energy=0.7, current_goal="获得权势"),
    }
    rels = {
        ("甄嬛", "皇后"):  0.2,
        ("皇后", "甄嬛"): -0.3,
    }
    return WorldState(tick=0, time_of_day="morning", weather="sunny",
                      characters=chars, relationships=rels)


def make_random_world(characters=None, seed: Optional[int] = None) -> WorldState:
    """Full world: randomized initial state."""
    if seed is not None:
        random.seed(seed)
    if characters is None:
        characters = CHARACTERS_FULL

    chars = {}
    for name in characters:
        chars[name] = CharacterState(
            location=random.choice(LOCATIONS),
            mood=random.choice(MOODS),
            energy=round(random.uniform(0.3, 1.0), 2),
            current_goal=random.choice(GOALS),
            knowledge=frozenset(random.sample(FACTS, k=random.randint(0, 3)))
        )

    rels = {}
    for a in characters:
        for b in characters:
            if a != b:
                rels[(a, b)] = round(random.uniform(-0.5, 0.5), 2)

    return WorldState(
        tick=0,
        time_of_day=random.choice(["morning", "afternoon", "evening", "night"]),
        weather=random.choice(["sunny", "rainy", "cloudy"]),
        characters=chars,
        active_events=frozenset(),
        relationships=rels
    )
