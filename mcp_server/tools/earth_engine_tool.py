# tools/earth_engine_tool.py

import asyncio
import json
from pathlib import Path
from langchain_core.tools import tool

API_URL = "https://my-backend-57y2ldgf7q-ew.a.run.app/analyze"

@tool(
    name="fetch_earth_engine_data",
    description="Fetch satellite analysis for a geo-point using longitude, latitude, start date, optional radius, and threshold factor.",
    args_schema={
        "lon": float,
        "lat": float,
        "recent_start": str,
        "radius": int,
        "thresholdFactor": float,
    }
)
async def fetch_earth_engine_data(
    lon: float,
    lat: float,
    recent_start: str,
    radius: int = 10,
    thresholdFactor: float = 2.5,
):
    import aiohttp
    from datetime import datetime

    payload = dict(
        lon=lon,
        lat=lat,
        recent_start=recent_start,
        radius=radius,
        thresholdFactor=thresholdFactor,
    )

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(API_URL, json=payload) as resp:
                if resp.status != 200:
                    api_result = {
                        "error": f"API returned {resp.status}",
                        "details": await resp.text(),
                    }
                else:
                    try:
                        api_result = await resp.json()
                    except Exception:
                        api_result = {
                            "error": "Non-JSON response",
                            "raw": await resp.text(),
                        }
        except Exception as e:
            api_result = {"error": str(e)}

    docs_dir = Path("/app/docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = docs_dir / f"geodata_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"input": payload, "api_response": api_result}, f, indent=2)

    build_script = "build_vectorstore.py"
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
