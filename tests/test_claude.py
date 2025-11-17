import pytest
import respx
from httpx import Response
import os
import sys
sys.path.append('..')
from app.main import generate_with_claude

@pytest.mark.asyncio
@respx.mock
async def test_claude_wrapper():
    mock_response = {
        "content": [{"text": "Hello world"}]
    }

    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(200, json=mock_response)
    )

    out = generate_with_claude("Test?")
    assert "Hello" in out
    assert route.called
