"""Pytest configuration for temporal-awareness tests."""

import sys
from pathlib import Path

import pytest

# Add project root to path so 'src' imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip slow tests (multi-architecture, benchmarks)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow (multi-architecture or benchmark)")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and command-line options."""
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="Skipped via --skip-slow")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
