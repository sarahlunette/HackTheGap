from fastmcp import FastMCP
from tools.earth_engine_tool import fetch_earth_engine_data

mcp = FastMCP("earth-engine-mcp")

@mcp.tool
async def fetch_earth_engine_data(
    lon: float,
    lat: float,
    recent_start: str,
    radius: int = 10,
    thresholdFactor: float = 2.5
):
    from tools.earth_engine_tool import fetch_earth_engine_data as real_impl
    return await real_impl(lon, lat, recent_start, radius, thresholdFactor)

app = mcp.http_app()