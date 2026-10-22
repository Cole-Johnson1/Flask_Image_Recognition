

"""Shared pytest fixtures."""
import pytest
from app import app


@pytest.fixture
def client():
    """Flask test client."""
    with app.test_client() as client:
        yield client
