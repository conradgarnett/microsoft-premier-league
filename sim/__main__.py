"""CLI: python -m sim --seed 7 --minutes 90 --out data/matches"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from .engine import MatchSimulator
from .skeleton import build_skeleton


def write_bundle(bundle, out_root: pathlib.Path, skeletons: bool = True) -> pathlib.Path:
    """Write a MatchBundle as plain JSON files: match.json, lowres.json, windows/, skeletons/."""
    d = out_root / bundle.match.meta.match_id
    (d / "windows").mkdir(parents=True, exist_ok=True)
    (d / "match.json").write_text(bundle.match.model_dump_json())
    (d / "lowres.json").write_text(bundle.lowres.model_dump_json())
    heights = {p.id: p.height_m for p in bundle.match.players()}
    vision = {p.id: p.attributes.vision for p in bundle.match.players()}
    for wid, w in bundle.windows.items():
        (d / "windows" / f"{wid}.json").write_text(w.model_dump_json())
        if skeletons:
            (d / "skeletons").mkdir(exist_ok=True)
            sk = build_skeleton(w, heights, vision)
            (d / "skeletons" / f"{wid}.json").write_text(json.dumps(sk.model_dump()))
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic football match")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--out", default="data/matches")
    ap.add_argument("--no-skeletons", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    bundle = MatchSimulator(seed=a.seed, duration_min=a.minutes).run()
    d = write_bundle(bundle, pathlib.Path(a.out), skeletons=not a.no_skeletons)
    m = bundle.match
    print(
        f"{m.meta.match_id}: {m.teams[0].name} {m.meta.final_score.home}-{m.meta.final_score.away} {m.teams[1].name}, "
        f"{len(m.events)} events, {len(bundle.windows)} replay windows -> {d} ({time.time() - t0:.1f}s)"
    )


if __name__ == "__main__":
    main()
