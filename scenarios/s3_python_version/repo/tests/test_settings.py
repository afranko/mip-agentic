from src.settings import get_timeout


def test_default_timeout():
    assert get_timeout() == 30
