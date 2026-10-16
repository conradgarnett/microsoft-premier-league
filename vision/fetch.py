"""Cache a short segment of a remote video locally (stream copy, no re-encode).

    python -m vision.fetch <video URL> <start_s> <duration_s> <out.mp4>

Later stages make several passes over a clip (detection, ball, review), so pulling the
segment once beats streaming it repeatedly. The copy starts at the keyframe before
``start_s``; ``<out>.json`` records the offset of the first frame so stages can address
source time. Keep the cache outside git.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import av


def fetch(url: str, start_s: float, duration_s: float, out: str) -> dict:
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with av.open(url, options={"timeout": "30000000"}) as src, av.open(out, "w") as dst:
        s = src.streams.video[0]
        o = dst.add_stream_from_template(s)
        src.seek(int(start_s / s.time_base), stream=s)  # lands on the keyframe at or before start
        first = None
        for pkt in src.demux(s):
            if pkt.pts is None:
                continue
            t = float(pkt.pts * s.time_base)
            if first is None:
                first = t
            if t > start_s + duration_s:
                break
            pkt.stream = o
            dst.mux(pkt)
    info = {"source_start_s": first, "requested_start_s": start_s, "duration_s": duration_s}
    pathlib.Path(out + ".json").write_text(json.dumps(info))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("start", type=float)
    ap.add_argument("duration", type=float)
    ap.add_argument("out")
    a = ap.parse_args()
    print(fetch(a.url, a.start, a.duration, a.out))


if __name__ == "__main__":
    main()
