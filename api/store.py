"""Match storage behind one small interface.

``LocalStore`` (default) keeps plain JSON files under ``DATA_DIR``. ``CosmosStore`` keeps
the same documents in Azure Cosmos DB when ``STORE=cosmos`` and ``COSMOS_ENDPOINT`` are set.
Heavy arrays (tracking windows, skeletons) stay as blobs: one document per window.
"""

from __future__ import annotations

import json
import os
import pathlib
from functools import lru_cache
from typing import Any, Protocol

from sim.__main__ import write_bundle
from sim.engine import MatchSimulator
from sim.schema import LowResTracking, Match, SkeletonWindow, TrackingWindow


class MatchStore(Protocol):
    def list_matches(self) -> list[str]: ...
    def get_doc(self, match_id: str, name: str) -> Any | None: ...
    def put_doc(self, match_id: str, name: str, doc: Any) -> None: ...


class LocalStore:
    def __init__(self, root: str | os.PathLike) -> None:
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, match_id: str, name: str) -> pathlib.Path:
        # trust boundary: ids come from URLs, so refuse anything that could escape the data dir
        for part in (match_id, *name.split("/")):
            if not part or part.startswith(".") or "\\" in part or not all(c.isalnum() or c in "-_." for c in part):
                raise KeyError(f"invalid id: {match_id}/{name}")
        return self.root / match_id / f"{name}.json"

    def list_matches(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "match.json").exists())

    def get_doc(self, match_id: str, name: str) -> Any | None:
        p = self._path(match_id, name)
        return json.loads(p.read_text()) if p.exists() else None

    def put_doc(self, match_id: str, name: str, doc: Any) -> None:
        p = self._path(match_id, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc))
        tmp.replace(p)  # atomic: never leave a half-written document


class CosmosStore:
    """Azure Cosmos DB (NoSQL API). Container partitioned by /match_id; id = "<match>:<name>".

    shortcut: Cosmos items max out at 2 MB, so larger documents (whole-match tracking,
    skeleton windows) go to ``blob_fallback`` with a pointer item in Cosmos; mount Azure
    Files there in production, or swap in Blob Storage at scale.
    """

    MAX_ITEM_BYTES = 1_500_000

    def __init__(self, endpoint: str, database: str, container: str, blob_fallback: LocalStore) -> None:
        from azure.cosmos import CosmosClient, PartitionKey
        from azure.identity import DefaultAzureCredential

        key = os.getenv("COSMOS_KEY")
        client = CosmosClient(endpoint, credential=key or DefaultAzureCredential())
        db = client.create_database_if_not_exists(database)
        self.c = db.create_container_if_not_exists(container, partition_key=PartitionKey(path="/match_id"))
        self.blobs = blob_fallback

    @staticmethod
    def _id(match_id: str, name: str) -> str:
        return f"{match_id}:{name.replace('/', '_')}"

    def list_matches(self) -> list[str]:
        q = "SELECT DISTINCT VALUE c.match_id FROM c WHERE c.name = 'match'"
        return sorted(self.c.query_items(q, enable_cross_partition_query=True))

    def get_doc(self, match_id: str, name: str) -> Any | None:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            item = self.c.read_item(self._id(match_id, name), partition_key=match_id)
        except CosmosResourceNotFoundError:
            return None
        return self.blobs.get_doc(match_id, name) if item.get("blob") else item["doc"]

    def put_doc(self, match_id: str, name: str, doc: Any) -> None:
        item = {"id": self._id(match_id, name), "match_id": match_id, "name": name}
        if len(json.dumps(doc)) > self.MAX_ITEM_BYTES:
            self.blobs.put_doc(match_id, name, doc)
            item["blob"] = True
        else:
            item["doc"] = doc
        self.c.upsert_item(item)


def make_store() -> MatchStore:
    local = LocalStore(os.getenv("DATA_DIR", "data/matches"))
    if os.getenv("STORE", "local") == "cosmos":
        return CosmosStore(os.environ["COSMOS_ENDPOINT"], os.getenv("COSMOS_DATABASE", "tactical"), os.getenv("COSMOS_CONTAINER", "matches"), local)
    return local


class MatchRepo:
    """Typed, cached access to match documents."""

    def __init__(self, store: MatchStore) -> None:
        self.store = store
        self.match = lru_cache(maxsize=8)(self._match)
        self.lowres = lru_cache(maxsize=4)(self._lowres)
        self.window = lru_cache(maxsize=32)(self._window)
        self.skeleton = lru_cache(maxsize=8)(self._skeleton)

    def _load(self, match_id: str, name: str) -> Any:
        doc = self.store.get_doc(match_id, name)
        if doc is None:
            raise KeyError(f"{match_id}/{name}")
        return doc

    def _match(self, match_id: str) -> Match:
        return Match.model_validate(self._load(match_id, "match"))

    def _lowres(self, match_id: str) -> LowResTracking:
        return LowResTracking.model_validate(self._load(match_id, "lowres"))

    def _window(self, match_id: str, wid: str) -> TrackingWindow:
        return TrackingWindow.model_validate(self._load(match_id, f"windows/{wid}"))

    def _skeleton(self, match_id: str, wid: str) -> SkeletonWindow:
        return SkeletonWindow.model_construct(**self._load(match_id, f"skeletons/{wid}"))

    def generate(self, seed: int, minutes: float) -> str:
        bundle = MatchSimulator(seed=seed, duration_min=minutes).run()
        if isinstance(self.store, LocalStore):
            import shutil

            shutil.rmtree(self.store.root / bundle.match.meta.match_id, ignore_errors=True)  # drop stale analytics/insights
            write_bundle(bundle, self.store.root)
        else:
            from sim.skeleton import build_skeleton

            mid = bundle.match.meta.match_id
            self.store.put_doc(mid, "match", bundle.match.model_dump(mode="json"))
            self.store.put_doc(mid, "lowres", bundle.lowres.model_dump(mode="json"))
            heights = {p.id: p.height_m for p in bundle.match.players()}
            vision = {p.id: p.attributes.vision for p in bundle.match.players()}
            for wid, w in bundle.windows.items():
                self.store.put_doc(mid, f"windows/{wid}", w.model_dump(mode="json"))
                self.store.put_doc(mid, f"skeletons/{wid}", build_skeleton(w, heights, vision).model_dump())
        for c in (self.match, self.lowres, self.window, self.skeleton):
            c.cache_clear()
        return bundle.match.meta.match_id
