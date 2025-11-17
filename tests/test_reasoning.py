import json
import pytest
import respx
from httpx import Response
import os
import sys
sys.path.append('..')

from app.main import generate_reasoning_with_mistral

@pytest.mark.asyncio
@respx.mock
async def test_reasoning_extracts_fields():
    # Mock Mistral API
    mock_json = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "intent": "geospatial_request",
                    "response_mode": "short",
                    "entities": {
                        "dataset": "LANDSAT",
                        "date": "2024-01-03",
                        "lon": 10.5,
                        "lat": -8.3,
                        "radius": 30
                    }
                })
            }
        }]
    }

    route = respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(200, json=mock_json)
    )

    result = generate_reasoning_with_mistral("Get Landsat at lon=10.5 lat=-8.3 on 2024-01-03")
    assert result["intent"] == "geospatial_request"
    assert result["entities"]["lon"] == 10.5
    assert result["entities"]["lat"] == -8.3
    assert result["entities"]["dataset"] == "LANDSAT"
    assert route.called
