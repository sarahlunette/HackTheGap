from fastmcp import FastMCP
import os
import sys
sys.path.append('.')

# Import async tool functions
from tools.osm_tool import run_osm_data_tool
from tools.climate_tool import run_climate_forecast_tool
from tools.earth_engine_tool import fetch_earth_engine_data

# Create FastMCP server instance
mcp = FastMCP(
    name="Resilience Crisis Tools",
    version="1.0.0"
)

# Register tools
mcp.add_tool(run_osm_data_tool)
mcp.add_tool(run_climate_forecast_tool)
mcp.add_tool(fetch_earth_engine_data)

if __name__ == "__main__":
    # Start an HTTP server on port 8000
    mcp.run()
