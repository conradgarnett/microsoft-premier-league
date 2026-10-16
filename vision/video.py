"""Frame access for local files or HTTP(S) URLs (PyAV seeks with range requests, no full download)."""

from __future__ import annotations

from collections.abc import Iterator

import av
import numpy as np


def iter_frames(src: str, start_s: float, duration_s: float, fps: float = 25.0, retries: int = 5) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t_seconds, RGB uint8 array) resampled to ``fps`` by nearest source timestamp.

    Remote streams drop now and then; on an I/O error the stream is reopened and resumed
    at the next target time, so long clips survive flaky connections.
    """
    targets = start_s + np.arange(int(round(duration_s * fps))) / fps
    k = 0
    for attempt in range(retries + 1):
        try:
            with av.open(src, options={"timeout": "30000000"}) as c:
                s = c.streams.video[0]
                s.thread_type = "AUTO"
                c.seek(int(max(targets[k] - 1, 0) / s.time_base), stream=s)
                prev = None
                for fr in c.decode(s):
                    t = float(fr.pts * s.time_base)
                    while k < len(targets) and t >= targets[k]:
                        # whichever of the previous/current frame is closer to the target time
                        pick = fr if prev is None or abs(t - targets[k]) <= abs(prev[0] - targets[k]) else prev[1]
                        yield float(targets[k]), pick.to_ndarray(format="rgb24")
                        k += 1
                    if k >= len(targets):
                        return
                    prev = (t, fr)
            return
        except av.error.OSError:
            if attempt == retries:
                raise
