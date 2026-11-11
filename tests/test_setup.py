"""Infrastructure smoke tests.

These check only that the development environment and package layout work.
No Tarneeb game logic is tested here; rule, scoring and engine tests belong to
Phase 1 and later (see project requirements).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tarneeb

EXPECTED_PYTHON = (3, 12)

SUBPACKAGES = (
    "core",
    "env",
    "agents",
    "models",
    "buffers",
    "training",
    "league",
    "evaluation",
    "app",
)

# tarneeb/core must stay a pure engine; see project requirements.
FORBIDDEN_IN_CORE = ("torch", "gymnasium", "pettingzoo", "streamlit", "fastapi")


def test_python_version_is_supported() -> None:
    assert sys.version_info[:2] == EXPECTED_PYTHON


def test_tarneeb_package_imports() -> None:
    assert tarneeb.__version__
    assert Path(tarneeb.__file__).name == "__init__.py"


def test_all_subpackages_import() -> None:
    import importlib

    for name in SUBPACKAGES:
        module = importlib.import_module(f"tarneeb.{name}")
        assert module.__doc__, f"tarneeb.{name} should document its role"


def test_core_does_not_import_heavy_dependencies() -> None:
    """Setup-level guard for the core-engine boundary.

    Runs in a fresh interpreter so that imports made by other tests cannot
    mask a violation.
    """
    code = (
        "import sys, importlib\n"
        "importlib.import_module('tarneeb.core')\n"
        f"forbidden = {FORBIDDEN_IN_CORE!r}\n"
        "leaked = sorted(m for m in forbidden if m in sys.modules)\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "", f"tarneeb.core imported: {result.stdout.strip()}"
