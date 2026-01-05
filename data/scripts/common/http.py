"""
HTTP utilities with retry logic for data fetching.
"""

import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def create_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """
    Create a requests session with retry logic.

    Args:
        retries: Number of retry attempts
        backoff: Backoff factor between retries

    Returns:
        Configured requests Session
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({"User-Agent": "can-i-run-data-updater/1.0"})

    return session


def fetch_json(url: str, session: requests.Session | None = None) -> dict[str, Any]:
    """
    Fetch JSON from URL with error handling.

    Args:
        url: URL to fetch
        session: Optional pre-configured session

    Returns:
        Parsed JSON as dictionary

    Raises:
        requests.RequestException: On network/HTTP errors
    """
    session = session or create_session()

    logger.debug(f"Fetching JSON: {url}")
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_text(url: str, session: requests.Session | None = None) -> str:
    """
    Fetch text content from URL with error handling.

    Args:
        url: URL to fetch
        session: Optional pre-configured session

    Returns:
        Response text content

    Raises:
        requests.RequestException: On network/HTTP errors
    """
    session = session or create_session()

    logger.debug(f"Fetching text: {url}")
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text
