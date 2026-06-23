from unittest.mock import patch
from fastapi.testclient import TestClient


# Mock the heavy AI imports before importing the app
@patch("interfaces.api.get_rag_chain")
@patch("interfaces.api.get_retriever")
@patch("interfaces.api.get_rewrite_chain")
def test_fastapi_boots_and_responds(mock_rewrite, mock_retriever, mock_chain):
    # Setup Mocks
    mock_rewrite.return_value.ainvoke.return_value = "mocked standalone question"
    mock_retriever.return_value.ainvoke.return_value = []
    mock_chain.return_value.astream.return_value = _async_generator(["Hello ", "World"])

    # Import app AFTER mocking
    from interfaces.api import app

    client = TestClient(app)

    # Test the /documents endpoint (doesn't require LLM)
    response = client.get("/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

    # Test the /metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "cpu_percent" in response.json()


# Helper for async streaming mock
async def _async_generator(items):
    for item in items:
        yield item
