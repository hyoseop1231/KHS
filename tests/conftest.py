import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    os.environ["DEBUG"] = "True"
    yield
    del os.environ["DEBUG"]