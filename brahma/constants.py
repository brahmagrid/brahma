"""Brahma — shared constants.

Import-safe module with no internal dependencies.  Import from anywhere.
Centralised so BRAHMA_HOME / skills-dir / config-path are defined once.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_brahma_home() -> Path:
    """Return the Brahma home directory (default: ~/.brahma).

    Reads ``BRAHMA_HOME`` env var; falls back to ``~/.brahma``.
    """
    val = os.environ.get("BRAHMA_HOME", "").strip()
    if val:
        return Path(val)
    return Path.home() / ".brahma"


def get_skills_dir() -> Path:
    """Return the skills directory under BRAHMA_HOME."""
    return get_brahma_home() / "skills"
