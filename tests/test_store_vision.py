import json

import numpy as np
import pytest

from api.store import CosmosStore, LocalStore


def test_local_store_rejects_traversal(tmp_path):
    s = LocalStore(tmp_path)
    s.put_doc("m1", "match", {"a": 1})
    assert s.get_doc("m1", "match") == {"a": 1}
    for bad in ("../x", "..", ".hidden"):
        with pytest.raises(KeyError):
            s.get_doc(bad, "match")
    with pytest.raises(KeyError):
        s.put_doc("m1", "../../etc/passwd", {})


def test_cosmos_store_routes_large_docs(tmp_path):
    """Items over the Cosmos size limit go to the blob fallback with a pointer item."""

    class FakeContainer:
        def __init__(self):
            self.items = {}

        def upsert_item(self, item):
            self.items[item["id"]] = item

        def read_item(self, id, partition_key):
            return self.items[id]

    store = CosmosStore.__new__(CosmosStore)
    store.c = FakeContainer()
    store.not_found = KeyError
    store.blobs = LocalStore(tmp_path)
    store.put_doc("m1", "match", {"small": True})
    big = {"x": "y" * (CosmosStore.MAX_ITEM_BYTES + 10)}
    store.put_doc("m1", "lowres", big)
    assert store.c.items["m1:match"]["doc"] == {"small": True}
    assert store.c.items["m1:lowres"]["blob"] is True and "doc" not in store.c.items["m1:lowres"]
    assert store.get_doc("m1", "lowres") == big


def test_calibration_round_trip():
    """Project pitch points to pixels and back for both camera models."""
    from vision.calibrate import PanoramaCamera

    for model in ("fisheye", "equirect"):
        p = {
            "model": model,
            "C": [50.0, -20.0, 12.0],
            "L": 100.0,
            "W": 64.0,
            "rows": {"top": [0, 1024], "bottom": [1024, 2048]},
            "top": {"rotvec": [-1.7, -0.6, 0.4], "f": 650.0, "k1": 0.05, "k2": 0.0, "cx": 1024.0, "cy": 512.0},
            "bottom": {"rotvec": [-1.8, 0.6, -0.45], "f": 650.0, "k1": 0.05, "k2": 0.0, "cx": 1024.0, "cy": 512.0},
        }
        if model == "equirect":
            for h in ("top", "bottom"):
                p[h]["f"] = 2048 / np.pi
        cam = PanoramaCamera(json.loads(json.dumps(p)))
        pts = np.array([[20.0, 30.0, 0], [35.0, 10.0, 0], [10.0, 50.0, 0]])
        uv = cam.to_pixels("top", pts)
        ok = np.isfinite(uv).all(1)
        assert ok.any()
        back = cam.to_pitch("top", uv[ok])
        assert np.allclose(back, pts[ok, :2], atol=0.05), (model, back, pts[ok])
