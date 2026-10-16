import pytest

from sim.engine import MatchSimulator


@pytest.fixture(scope="session")
def bundle():
    # 20 simulated minutes: enough for shots and replay windows, fast enough for CI
    return MatchSimulator(seed=3, duration_min=20).run()
