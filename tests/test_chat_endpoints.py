import json
import pytest
import respx
from httpx import Response
from fastapi.testclient import TestClient
import os
import sys
sys.path.append('..')
from app.main import app

client = TestClient(app)

@pytest.mark.asyncio
@respx.mock
async def test_chat_endpoint_with_mocked_services():

    # --- Mock Mistral Reasoning
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(200, json={
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "intent": "simple_question",
                        "response_mode": "short",
                        "entities": {}
                    })
                }
            }]
        })
    )

    # --- Mock Claude
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(200, json={"content": [{"text": "Test reply"}]})
    )

    response = client.post(
        "/chat",
        auth=("admin", "password"),
        json={"question": "Hello?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "Test reply" in data["answer"]
