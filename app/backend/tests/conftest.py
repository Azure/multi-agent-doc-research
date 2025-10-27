import pytest
import os
import sys

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Settings


@pytest.fixture
def settings():
    """Load settings for tests."""
    return Settings()
