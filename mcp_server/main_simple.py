from fastmcp import FastMCP

# Import async wrappers for existing tools (commented out for now)
# from tools.osm_tool import run_osm_data_tool
# from tools.climate_tool import run_climate_forecast_tool
# from tools.earth_engine_tool import fetch_earth_engine_data

# Create FastMCP server
mcp = FastMCP("Resilience Crisis Tools", version="1.0.0")

# Register tools (commented out for now)
# mcp.add_tool(run_osm_data_tool)
# mcp.add_tool(run_climate_forecast_tool)
# mcp.add_tool(fetch_earth_engine_data)

# Simple health check tool for testing
@mcp.tool()
def health_check() -> str:
    """Simple health check tool"""
    return "MCP server is healthy"

if __name__ == "__main__":
    # Run server with SSE transport for HTTP/WebSocket access
    mcp.run(transport="sse")
