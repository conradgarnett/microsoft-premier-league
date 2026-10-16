"""Fictional clubs and players. Every name here is invented for the synthetic league."""

from __future__ import annotations

import numpy as np

from .schema import Attributes, Player, Position, Team

# Base shapes in the team frame (team attacks towards +x, own goal at x=0).
# Facing +x, the team's right flank is low y.
FORMATIONS: dict[str, list[tuple[Position, float, float]]] = {
    "4-3-3": [
        (Position.GK, 5.0, 34.0),
        (Position.RB, 30.0, 10.0),
        (Position.CB, 24.0, 26.0),
        (Position.CB, 24.0, 42.0),
        (Position.LB, 30.0, 58.0),
        (Position.DM, 40.0, 34.0),
        (Position.CM, 51.0, 22.0),
        (Position.CM, 51.0, 46.0),
        (Position.RW, 68.0, 9.0),
        (Position.ST, 73.0, 34.0),
        (Position.LW, 68.0, 59.0),
    ],
    "4-4-2": [
        (Position.GK, 5.0, 34.0),
        (Position.RB, 28.0, 10.0),
        (Position.CB, 23.0, 26.0),
        (Position.CB, 23.0, 42.0),
        (Position.LB, 28.0, 58.0),
        (Position.RM, 50.0, 10.0),
        (Position.CM, 45.0, 27.0),
        (Position.CM, 45.0, 41.0),
        (Position.LM, 50.0, 58.0),
        (Position.ST, 70.0, 28.0),
        (Position.ST, 70.0, 40.0),
    ],
}

_ROSTERS = {
    "KBY": {
        "name": "Kestrel Bay FC",
        "short": "KBY",
        "colors": ("#2f81f7", "#e6f0ff"),
        "formation": "4-3-3",
        "press": 0.75,
        "players": [
            (1, "Tomas Varga", "right"),
            (2, "Eli Okonkwo", "right"),
            (4, "Marco Lindqvist", "right"),
            (5, "Dario Mensah", "left"),
            (3, "Felix Arnaud", "left"),
            (6, "Kenji Morales", "right"),
            (8, "Luca Brandt", "right"),
            (10, "Ilyas Coelho", "left"),
            (7, "Ruben Achterberg", "left"),
            (9, "Samuel Nakamura-Reyes", "right"),
            (11, "Jonah Petrescu", "right"),
        ],
    },
    "IRV": {
        "name": "Ironvale Athletic",
        "short": "IRV",
        "colors": ("#f2683c", "#2a160d"),
        "formation": "4-4-2",
        "press": 0.45,
        "players": [
            (1, "Aurelio Kask", "right"),
            (2, "Mateo Ferreira-Lund", "right"),
            (5, "Oskar Haddad", "right"),
            (4, "Callum Ibekwe", "left"),
            (3, "Nils Sorvik", "left"),
            (7, "Teo Valdivia", "right"),
            (8, "Ari Kowalczyk", "right"),
            (6, "Bastien Moreau-Diallo", "right"),
            (11, "Yusuf Delacorte", "left"),
            (9, "Viktor Amarante", "right"),
            (10, "Emil Castellano-Voss", "left"),
        ],
    },
}

# Attribute templates per position: (pace, passing, shooting, dribbling, tackling, vision, gk)
_POS_ATTRS = {
    Position.GK: (55, 62, 20, 30, 30, 55, 80),
    Position.RB: (80, 70, 45, 68, 72, 62, 10),
    Position.LB: (80, 70, 45, 68, 72, 62, 10),
    Position.CB: (68, 66, 40, 55, 80, 58, 10),
    Position.DM: (68, 78, 55, 68, 76, 76, 10),
    Position.CM: (72, 80, 66, 74, 64, 80, 10),
    Position.RM: (82, 74, 64, 78, 50, 72, 10),
    Position.LM: (82, 74, 64, 78, 50, 72, 10),
    Position.RW: (88, 74, 74, 84, 40, 74, 10),
    Position.LW: (88, 74, 74, 84, 40, 74, 10),
    Position.ST: (84, 68, 84, 78, 38, 70, 10),
}


def _short(name: str) -> str:
    parts = name.split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def build_team(code: str, seed: int = 0) -> Team:
    r = _ROSTERS[code]
    rng = np.random.default_rng(sum(ord(c) * 31**i for i, c in enumerate(code)) + seed)
    shape = FORMATIONS[r["formation"]]
    players = []
    for (number, name, foot), (pos, _, _) in zip(r["players"], shape, strict=True):
        base = _POS_ATTRS[pos]
        jit = rng.integers(-6, 7, size=7)
        pace, passing, shooting, dribbling, tackling, vision, gk = (int(np.clip(b + j, 25, 95)) for b, j in zip(base, jit, strict=True))
        players.append(
            Player(
                id=f"{code.lower()}-{number:02d}",
                name=name,
                short_name=_short(name),
                number=number,
                position=pos,
                team_id=code.lower(),
                foot=foot,
                height_m=round(float(rng.normal(1.82 if pos in (Position.GK, Position.CB) else 1.78, 0.04)), 2),
                attributes=Attributes(
                    pace=pace,
                    passing=passing,
                    shooting=shooting,
                    dribbling=dribbling,
                    tackling=tackling,
                    vision=vision,
                    stamina=int(rng.integers(65, 92)),
                    composure=int(rng.integers(55, 90)),
                    goalkeeping=gk,
                ),
            )
        )
    return Team(
        id=code.lower(),
        name=r["name"],
        short_name=r["short"],
        color_primary=r["colors"][0],
        color_secondary=r["colors"][1],
        formation=r["formation"],
        press_intensity=r["press"],
        players=players,
    )


def default_teams(seed: int = 0) -> tuple[Team, Team]:
    return build_team("KBY", seed), build_team("IRV", seed)
