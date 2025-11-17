from fastmcp import FastMCP

# Import async tool functions
from tools.osm_tool import run_osm_data_tool
from tools.climate_tool import run_climate_forecast_tool
from tools.earth_engine_tool import fetch_earth_engine_data

# Create FastMCP server instance
mcp = FastMCP(
    name="Resilience Crisis Tools",
    version="1.0.0"
)

# Register tools (no decorators needed in v0.3.x)
mcp.add_tool(run_osm_data_tool)
mcp.add_tool(run_climate_forecast_tool)
mcp.add_tool(fetch_earth_engine_data)

if __name__ == "__main__":
    # SSE transport works with VS Code, Claude, Cursor, etc.
    mcp.run(transport="sse")
