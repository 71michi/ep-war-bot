"""OpenAI client factory.

This module exists to reduce import-time side effects and to improve testability.
The rest of the code should call :func:`get_openai_client` instead of instantiating
`OpenAI()` directly at import time.

In unit tests you can monkeypatch `get_openai_client()` to return a stub.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_openai_client() -> Any:
    """Return a singleton OpenAI client.

    Import is done lazily to keep module import side effects small and to make
    lightweight unit tests possible even before dependencies are installed.
    """
    from openai import OpenAI  # local import (lazy)
    return OpenAI()
