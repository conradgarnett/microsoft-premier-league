import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from agents.orchestrator import Orchestrator, ValidationError, check_evidence
from agents.tools import MatchTools
from api.store import LocalStore, MatchRepo
from sim.__main__ import write_bundle


@pytest.fixture(scope="session")
def repo(bundle, tmp_path_factory):
    root = tmp_path_factory.mktemp("matches")
    write_bundle(bundle, root)
    return MatchRepo(LocalStore(root))


@pytest.fixture(scope="session")
def tools(repo):
    return MatchTools(repo)


@pytest.fixture(scope="session")
def mid(bundle):
    return bundle.match.meta.match_id


def test_evidence_validator():
    check_evidence({"claims": [{"text": "x", "evidence": ["a"]}]}, {"a"})
    with pytest.raises(ValidationError):
        check_evidence({"text": "x", "evidence": []}, {"a"})
    with pytest.raises(ValidationError):
        check_evidence([{"text": "x", "evidence": ["zzz"]}], {"a"})


@pytest.mark.parametrize("mode,lang", [("analyst", "en"), ("fan", "es"), ("analyst", "fr")])
def test_orchestrator_offline(tools, mid, mode, lang):
    out = asyncio.run(Orchestrator(tools).run(mid, mode, lang, None))
    assert out["provider"] == "offline"
    valid = tools.valid_event_ids(mid)
    check_evidence(out["moments"], valid)
    check_evidence(out["recap"], valid)
    assert out["director"]["reel"]
    assert len(out["moments"]) == len(tools.report(mid)["key_moments"])
    agents = {t["agent"] for t in out["trace"]}
    assert {"Replay Director", "Tactical Analyst", "Commentator", "Recap Writer"} <= agents
    if lang != "en":
        assert "Translator" in agents
    assert all(t["status"] == "ok" for t in out["trace"])


def test_orchestrator_falls_back_when_model_misbehaves(tools, mid):
    """A model that cites fake evidence is retried, then replaced by the offline agent."""

    class BadAgent:
        async def run(self, prompt):
            class R:
                text = json.dumps({"reel": [{"text": "made up", "evidence": ["ev99999"]}]})

            return R()

    o = Orchestrator(tools)
    o.provider = "azure-openai"
    o.agents["director"] = BadAgent()
    from agents.orchestrator import MatchState

    state = MatchState(mid, "analyst", "en", None, tools.valid_event_ids(mid))
    out = asyncio.run(o.call(state, "director", "director", {"moments": tools.report(mid)["key_moments"]}))
    assert out["reel"]
    assert [t["status"] for t in state.trace] == ["rejected", "rejected", "ok"]
    assert state.trace[-1]["provider"] == "offline"


def test_api_endpoints(repo, tools, mid, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "repo", repo)
    monkeypatch.setattr(main, "tools", tools)
    monkeypatch.setattr(main, "orchestrator", Orchestrator(tools))
    c = TestClient(main.app)
    assert c.get("/api/health").json()["status"] == "ok"
    assert c.get("/api/matches").json()[0]["match_id"] == mid
    km = c.get(f"/api/matches/{mid}/key-moments").json()
    assert km and all(k["evidence"] for k in km)
    r = c.get(f"/api/matches/{mid}/replay/{km[0]['window_id']}").json()
    assert len(r["skeleton"]["frames"]) == r["tracking"]["n_frames"]
    assert "teams" in c.get(f"/api/matches/{mid}/metrics").json()
    ins = c.get(f"/api/matches/{mid}/insights", params={"mode": "fan", "lang": "pt"}).json()
    assert ins["lang"] == "pt" and ins["recap"]["claims"]
    assert c.get(f"/api/matches/{mid}/insights", params={"lang": "xx"}).status_code == 422
    assert c.get("/api/matches/..%2F..%2Fetc/events").status_code == 404
    with c.websocket_connect(f"/api/matches/{mid}/live?speed=120") as ws:
        kinds = {ws.receive_json()["type"] for _ in range(30)}
    assert "frame" in kinds


def test_mcp_tools(tools, mid):
    from mcp import Client

    from mcp_server.server import build_server

    async def go():
        async with Client(build_server(tools)) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert {"get_events", "get_key_moments", "get_player_stats", "get_skeleton_window", "get_pressure_map"} <= names
            r = await c.call_tool("get_events", {"match_id": mid, "types": ["shot"], "limit": 3})
            assert not r.is_error
            bad = await c.call_tool("get_player_stats", {"match_id": mid, "player_id": "nobody"})
            assert bad.is_error

    asyncio.run(go())
