"""Source-agnostic match data schema.

These pydantic models describe everything downstream modules consume: match metadata,
rosters, the event stream, positional tracking and 3D skeleton keypoints. The synthetic
engine in :mod:`sim.engine` is one producer; a future video-to-skeleton module is meant to
produce exactly the same shapes (``source`` tells consumers where data came from and
``confidence`` lets noisy producers attach per-joint certainty).

Coordinate system (all producers):
    * Pitch is 105 m x 68 m. ``x`` runs along the length (0 = left goal line,
      105 = right goal line), ``y`` runs across (0 = bottom touchline, 68 = top).
    * ``z`` is height above the grass in metres.
    * Time ``t`` is seconds since kick-off of the match (continuous across halves).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44
GOAL_Y_LOW = (PITCH_WIDTH - GOAL_WIDTH) / 2  # 30.34
GOAL_Y_HIGH = (PITCH_WIDTH + GOAL_WIDTH) / 2  # 37.66

SCHEMA_VERSION = "1.0.0"

# 17-joint skeleton. Order is part of the contract.
JOINT_NAMES: list[str] = [
    "head",
    "neck",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "pelvis",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
    "l_toe",
    "r_toe",
]
J = {name: i for i, name in enumerate(JOINT_NAMES)}

# Bones (pairs of joint indices) used by renderers.
BONES: list[tuple[str, str]] = [
    ("head", "neck"),
    ("neck", "l_shoulder"),
    ("neck", "r_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("neck", "pelvis"),
    ("pelvis", "l_hip"),
    ("pelvis", "r_hip"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("l_ankle", "l_toe"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
    ("r_ankle", "r_toe"),
]


class Position(str, Enum):
    GK = "GK"
    RB = "RB"
    CB = "CB"
    LB = "LB"
    DM = "DM"
    CM = "CM"
    RM = "RM"
    LM = "LM"
    RW = "RW"
    LW = "LW"
    ST = "ST"


class Attributes(BaseModel):
    """0-100 ratings that drive the synthetic engine. Optional for other sources."""

    pace: int = 70
    passing: int = 70
    shooting: int = 60
    dribbling: int = 65
    tackling: int = 60
    vision: int = 65
    stamina: int = 75
    composure: int = 65
    goalkeeping: int = 10


class Player(BaseModel):
    id: str
    name: str
    short_name: str
    number: int
    position: Position
    team_id: str
    foot: Literal["left", "right"] = "right"
    height_m: float = 1.8
    attributes: Attributes = Field(default_factory=Attributes)


class Team(BaseModel):
    id: str
    name: str
    short_name: str
    color_primary: str
    color_secondary: str
    formation: str
    press_intensity: float = Field(0.5, ge=0, le=1, description="0 = low block, 1 = full press")
    players: list[Player]


class EventType(str, Enum):
    KICKOFF = "kickoff"
    PASS = "pass"
    CARRY = "carry"
    SHOT = "shot"
    GOAL = "goal"
    SAVE = "save"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    PRESSURE = "pressure"
    FOUL = "foul"
    FREE_KICK = "free_kick"
    GOAL_KICK = "goal_kick"
    CLEARANCE = "clearance"
    BALL_RECOVERY = "ball_recovery"
    POSSESSION_CHANGE = "possession_change"
    HALF_START = "half_start"
    HALF_END = "half_end"


class Event(BaseModel):
    """One discrete on-ball (or game-state) event."""

    id: str = Field(description="Stable id, e.g. ev00042. Agents cite these as evidence.")
    t: float = Field(description="Seconds since match kick-off")
    period: int
    minute: int
    second: int
    type: EventType
    team_id: str | None = None
    player_id: str | None = None
    x: float | None = None
    y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    outcome: str | None = Field(None, description="complete/incomplete, won/lost, goal/saved/off_target/blocked ...")
    receiver_id: str | None = None
    opponent_id: str | None = None
    possession_id: int = 0
    under_pressure: bool = False
    related_ids: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class TrackingWindow(BaseModel):
    """Columnar positional tracking for a time window.

    ``players[f][p]`` is ``[x, y]`` for frame ``f`` and the ``p``-th id in ``player_ids``.
    ``ball[f]`` is ``[x, y, z]``.
    """

    id: str
    source: str = "synthetic"
    t_start: float
    fps: float
    n_frames: int
    player_ids: list[str]
    ball: list[list[float]]
    players: list[list[list[float]]]
    actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Body actions in the window: {t, player_id, kind: kick|tackle|dive|receive, ...}",
    )
    event_ids: list[str] = Field(default_factory=list)


class SkeletonWindow(BaseModel):
    """3D keypoints. ``frames[f][p][j]`` is ``[x, y, z]`` in pitch metres."""

    id: str
    window_id: str
    source: str = "synthetic"
    t_start: float
    fps: float
    n_frames: int
    joint_names: list[str] = Field(default_factory=lambda: list(JOINT_NAMES))
    bones: list[tuple[str, str]] = Field(default_factory=lambda: list(BONES))
    player_ids: list[str]
    frames: list[list[list[list[float]]]]
    confidence: list[list[list[float]]] | None = Field(
        None, description="Optional per-joint confidence [f][p][j] in 0..1 (e.g. from a pose model)"
    )


class Score(BaseModel):
    home: int = 0
    away: int = 0


class MatchMeta(BaseModel):
    schema_version: str = SCHEMA_VERSION
    match_id: str
    source: str = "synthetic"
    seed: int | None = None
    competition: str = "Inside the Game Synthetic League"
    venue: str = "Harborview Park"
    date: str = "2026-10-24"
    duration_s: float
    half_length_s: float
    pitch_length: float = PITCH_LENGTH
    pitch_width: float = PITCH_WIDTH
    fps: float = 25.0
    home_team_id: str
    away_team_id: str
    home_attacks_right_first_half: bool = True
    final_score: Score = Field(default_factory=Score)


class WindowRef(BaseModel):
    id: str
    t_start: float
    t_end: float
    trigger_event_ids: list[str]


class Match(BaseModel):
    """Everything except the heavy per-window arrays."""

    meta: MatchMeta
    teams: list[Team]
    events: list[Event]
    windows: list[WindowRef] = Field(default_factory=list)

    def team(self, team_id: str) -> Team:
        return next(t for t in self.teams if t.id == team_id)

    def player(self, player_id: str) -> Player:
        for t in self.teams:
            for p in t.players:
                if p.id == player_id:
                    return p
        raise KeyError(player_id)

    def players(self) -> list[Player]:
        return [p for t in self.teams for p in t.players]

    def attack_direction(self, team_id: str, period: int) -> int:
        """+1 if ``team_id`` attacks towards x=105 in ``period``, else -1."""
        home_first = 1 if self.meta.home_attacks_right_first_half else -1
        d = home_first if team_id == self.meta.home_team_id else -home_first
        return d if period == 1 else -d


class LowResTracking(BaseModel):
    """Whole-match tracking at a low frame rate (default 2 fps) for match-level analytics."""

    source: str = "synthetic"
    fps: float
    t_start: float = 0.0
    n_frames: int
    player_ids: list[str]
    ball: list[list[float]]
    players: list[list[list[float]]]


class MatchBundle(BaseModel):
    match: Match
    lowres: LowResTracking
    windows: dict[str, TrackingWindow] = Field(default_factory=dict)


def json_schemas() -> dict[str, Any]:
    """JSON Schemas for every public model (written to docs/schema/ by `python -m sim.schema`)."""
    return {
        m.__name__: m.model_json_schema()
        for m in (Match, Event, TrackingWindow, SkeletonWindow, LowResTracking, Team, Player)
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    import pathlib

    out = pathlib.Path("docs/schema")
    out.mkdir(parents=True, exist_ok=True)
    for name, schema in json_schemas().items():
        (out / f"{name}.schema.json").write_text(json.dumps(schema, indent=2))
    print(f"wrote {len(json_schemas())} schemas to {out}")
