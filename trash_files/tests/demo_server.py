from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI(title="Public Demo MCP")

def generate_md5_hash(input_str: str) -> str:
    import hashlib
    md5_hash = hashlib.md5()
    md5_hash.update(input_str.encode('utf-8'))
    return md5_hash.hexdigest()

def count_characters(input_str: str) -> int:
    return len(input_str)

def get_first_half(input_str: str) -> str:
    midpoint = len(input_str) // 2
    return input_str[:midpoint]

# MCP tool endpoint routing
@app.post("/generate_md5_hash")
def md5_route(input_str: str):
    return {"result": generate_md5_hash(input_str)}

@app.post("/count_characters")
def count_route(input_str: str):
    return {"result": count_characters(input_str)}

@app.post("/get_first_half")
def first_half_route(input_str: str):
    return {"result": get_first_half(input_str)}

# Expose all tools as MCP via FastApiMCP
mcp = FastApiMCP(app)
mcp.mount()

# To launch: uvicorn demo_server:app --port 9001 --reload
