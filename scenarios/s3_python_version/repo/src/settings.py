DEFAULT_TIMEOUT = 30


def get_timeout(value: int | None = None) -> int:
    """Return the request timeout in seconds."""
    return value if value is not None else DEFAULT_TIMEOUT
