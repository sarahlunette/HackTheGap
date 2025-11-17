import aiohttp
import asyncio

MCP_URL = "http://mcp_server:9001/tools/fetch_earth_engine_data"


async def test_call_mcp_tool():
    payload = {
        "lon": 14.5,
        "lat": -22.1,
        "recent_start": "2025-10-01",
        "radius": 25,
        "thresholdFactor": 2.5
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(MCP_URL, json=payload) as resp:
            print("Status:", resp.status)
            text = await resp.text()
            print("Response:", text)


if __name__ == "__main__":
    asyncio.run(test_call_mcp_tool())
