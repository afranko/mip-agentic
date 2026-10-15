from src.http_client import fetch_status


def test_fetch_status_is_callable():
    assert callable(fetch_status)
