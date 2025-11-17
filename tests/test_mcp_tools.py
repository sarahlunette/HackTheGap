import pytest
import respx
from pathlib import Path
from httpx import Response
import os
import sys
sys.path.append('..')
from mcp_server.tools.earth_engine_tool import fetch_earth_engine_data


@pytest.mark.asyncio
@respx.mock
async def test_mcp_tool(tmp_path, monkeypatch):

    # --- mock API
    respx.post("https://my-backend-57y2ldgf7q-ew.a.run.app/analyze").mock(
        return_value=Response(200, json={"ok": True})
    )

    # --- mock vectorstore rebuild
    async def fake_subprocess(*args, **kwargs):
        class P:
            async def communicate(self_inner):
                return (b"rebuild ok", b"")
        return P()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_subprocess)

    # --- override docs folder
    monkeypatch.setattr("mcp_server.tools.earth_engine_tool.Path", lambda p: tmp_path / "docs")

    result = await fetch_earth_engine_data(
        lon= 14.5, lat= -22.1, recent_start= '2025-10-01', radius= 25
    )

    print(result)

    # Check JSON saved
    files = list((tmp_path / "docs").glob("*.json"))
    assert len(files) == 1
    #assert result["api_response"]["ok"] is True
    assert "rebuild ok" in result["vectorstore_update_log"]
