import os

import pytest

os.environ["LLM_MODE"] = "mock"          # the smoke tests never call the real API

from agentlab.core import Scenario, Trace  # noqa: E402
from agentlab.llm import MockLLM  # noqa: E402


@pytest.fixture
def scenario():
    return lambda name: Scenario.load(name)


@pytest.fixture
def trace(tmp_path):
    return lambda name="t", approach="test": Trace(name, approach, out_dir=tmp_path)


@pytest.fixture
def mock():
    return lambda persona="good": MockLLM(persona=persona)
