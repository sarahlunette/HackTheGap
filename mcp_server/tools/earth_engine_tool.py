import asyncio
import json
from pathlib import Path

API_URL = "https://my-backend-57y2ldgf7q-ew.a.run.app/analyze"


# ------------------------------------------------------------
# FastMCP 0.3.x — NO decorators inside tool files
# Tools are registered in main.py with mcp.add_tool()
# ------------------------------------------------------------
async def fetch_earth_engine_data(
    lon: float,
    lat: float,
    recent_start: str,
    radius: int = 10,
    thresholdFactor: float = 2.5
):
    """
    Calls your backend ML API with geospatial parameters.

    Args:
        lon (float)
        lat (float)
        recent_start (str): ISO date "YYYY-MM-DD"
        radius (int): optional
        thresholdFactor (float): optional

    Returns:
        dict: saved file info + backend response + vectorstore rebuild logs
    """

    import aiohttp
    from datetime import datetime

    # ------------------------------------------------------------
    # 1. Build payload for your backend API
    # ------------------------------------------------------------
    payload = {
        "lon": lon,
        "lat": lat,
        "recent_start": recent_start,
        "radius": radius,
        "thresholdFactor": thresholdFactor,
    }

    # ------------------------------------------------------------
    # 2. Call external backend API
    # ------------------------------------------------------------
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(API_URL, json=payload) as resp:
                if resp.status != 200:
                    api_result = {
                        "error": f"API returned {resp.status}",
                        "details": await resp.text()
                    }
                else:
                    # More robust json decoding
                    try:
                        api_result = await resp.json()
                    except:
                        api_result = {
                            "error": "Non-JSON response",
                            "raw": await resp.text()
                        }
        except Exception as e:
            api_result = {"error": str(e)}

    # ------------------------------------------------------------
    # 3. Save JSON to app/docs/ for vectorstore ingestion
    # ------------------------------------------------------------
    docs_dir = Path("/docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = docs_dir / f"geodata_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {"input": payload, "api_response": api_result},
            f,
            indent=2
        )

    # ------------------------------------------------------------
    # 4. Trigger vectorstore rebuild
    # ------------------------------------------------------------
    build_script = "app/build_vectorstore.py"

    try:
        proc = await asyncio.create_subprocess_exec(
            "python", build_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        rebuild_log = stdout.decode() + "\n" + stderr.decode()
    except Exception as e:
        rebuild_log = f"Error running build_vectorstore.py: {e}"

    # ------------------------------------------------------------
    # 5. Return result to MCP user
    # ------------------------------------------------------------
    return {
        "saved_to": str(filename),
        "api_response": api_result,
        "lon": lon,
        "lat": lat,
        "recent_start": recent_start,
        "radius": radius,
        "thresholdFactor": thresholdFactor,
        "vectorstore_update_log": rebuild_log,
        "message": "Data saved and vectorstore updated."
    }

# Test feature
if __name__ == "__main__":
    print("🔧 Running earth_engine_tool standalone test...\n")

    async def test():
        result = await fetch_earth_engine_data(
            lon=14.5,
            lat=-22.1,
            recent_start="2025-10-01",
            radius=25,
        )
        print("\n=== RESULT ===")
        print(list(result.keys()))
        output_path = "api_response_test.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result["api_response"], f, indent=2)

        print(f"Saved api_response → {output_path}")

    asyncio.run(test())