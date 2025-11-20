import os
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

# Importer les fonctions outils depuis tes scripts tools/
from tools.earth_engine_tool import earth_engine_tool
from tools.climate_tool import climate_tool


app = FastAPI(title="Crisis MCP API")
"""
FASTAPI MVP — RAG + Reasoning Model (Mistral) + Claude Sonnet 4.5 + MCP Tools
Now supporting async MCP Earth Engine data ingestion directly from /chat.
"""

import os
import datetime
import logging
import json
from pathlib import Path
from collections import defaultdict

import requests
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Anthropic Claude
from anthropic import Anthropic
from mistralai import Mistral

# RAG (Qdrant + LlamaIndex)
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import asyncio
from fastmcp import Client

# =========================
# ENV & CONFIG
# =========================
load_dotenv()

AUTH_MODE = os.getenv("AUTH_MODE", "basic")
MVP_USER = os.getenv("MVP_USER", "admin")
MVP_PASS = os.getenv("MVP_PASS", "password")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "island_docs")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mvp")

ACTION_LOGS: list[dict] = []

# =========================
# RAG
# =========================
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
qdrant_client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)
query_engine = index.as_retriever(similarity_top_k=3)

def query_knowledge_base(question: str) -> str:
    try:
        nodes = query_engine.retrieve(question)
    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        return ""
    return "\n".join(n.text for n in nodes)

# =========================
# MCP Example tools
# =========================
def addition(a: int, b: int) -> int:
    return a + b

def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny"

async def call_mcp_fetch_earth_engine(lon, lat, date, radius):
    payload = {
        "lon": float(lon),
        "lat": float(lat),
        "recent_start": str(date),
        "radius": int(radius)
    }
    try:
        client = Client("http://localhost:9001/mcp")
        async with client:
            result = await client.call_tool("fetch_earth_engine_data", payload)
            return result
    except Exception as e:
        return {"error": f"MCP client call failed: {str(e)}"}

# =========================
# Prompt blocks (reasoning, etc)
# =========================
REASONING_PROMPT = """
You are a reasoning model responsible for extracting structured parameters from
the user's message so the crisis-resilience assistant can decide whether to:
A) generate a narrative/plan, OR
B) call the geospatial MCP tool fetch_earth_engine_data.
(output JSON structure as specified)
User message:
"{user_question}"
"""

def extract_json_block(text: str) -> str:
    cleaned = text.replace("``````", "").strip()
    brace_level = 0; start = None
    for i, char in enumerate(cleaned):
        if char == "{":
            if brace_level == 0: start = i
            brace_level += 1
        elif char == "}":
            brace_level -= 1
            if brace_level == 0 and start is not None:
                return cleaned[start:i + 1].strip()
    raise ValueError(f"No JSON object found in text:\n{cleaned}")

def mistral_call_with_retry(prompt, model="open-mistral-nemo", retries=5):
    client = Mistral(api_key=MISTRAL_API_KEY)
    for i in range(retries):
        try:
            res = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500,
            )
            return res.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "capacity" in str(e):
                wait = 2 ** i
                print(f"Rate-limited — retrying in {wait}s...")
                time.sleep(wait); continue
            raise
    raise RuntimeError("Mistral: too many retries")

def generate_reasoning_with_mistral(user_question: str) -> dict:
    if not MISTRAL_API_KEY:
        logger.warning("MISTRAL_API_KEY missing — fallback mode.")
        return {}
    client = Mistral(api_key=MISTRAL_API_KEY)
    prompt = REASONING_PROMPT.replace("{user_question}", user_question)
    try:
        res = client.chat.complete(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000,
        )
        raw = res.choices[0].message.content.strip()
        json_text = extract_json_block(raw)
        result = json.loads(json_text)
    except Exception as e:
        logger.error(f"[Reasoning] Mistral SDK error: {e}")
        return {}
    return result

def generate_with_claude(prompt: str) -> str:
    with anthropic_client.messages.stream(
        model=CLAUDE_MODEL,
        temperature=0.7,
        max_tokens=64000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        return stream.get_final_text()

# =========================
# FastAPI TCP Server
# =========================

security = HTTPBasic()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def verify_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
    authorization: str = Header(None),
):
    if credentials.username != MVP_USER or credentials.password != MVP_PASS:
        raise HTTPException(401, "Unauthorized")
    return credentials.username

class ChatRequest(BaseModel):
    question: str
    tool: str = None
    args: dict = {}

@app.post("/chat")
async def chat(req: ChatRequest, username: str = Depends(verify_credentials)):
    memory = defaultdict(list)
    user_msg = req.question.strip()

    reasoning_output = generate_reasoning_with_mistral(user_msg)
    rag_context = query_knowledge_base(user_msg)

    # MCP tool routing + function calling (dynamic)
    result = None
    geospatial_result = None
    if req.tool == "addition":
        result = addition(**req.args)
    elif req.tool == "get_weather":
        result = get_weather(**req.args)
    elif (
        reasoning_output.get("intent") == "geospatial_request"
        and reasoning_output.get("entities", {}).get("lon") is not None
        and reasoning_output.get("entities", {}).get("lat") is not None
        and reasoning_output.get("entities", {}).get("date") is not None
    ):
        entities = reasoning_output["entities"]
        geospatial_result = await call_mcp_fetch_earth_engine(
            lon=entities["lon"],
            lat=entities["lat"],
            date=entities["date"],
            radius=int(entities.get("radius") or 10)
        )
        result = geospatial_result
    else:
        # LLM (Claude, Mistral) fallback generation
        prompt = f"RAG CONTEXT: {rag_context}\nUSER MESSAGE: {user_msg}"
        result = generate_with_claude(prompt)

    # Optionally, memory and logs
    memory[username].append({"user": user_msg, "ai": result})
    ACTION_LOGS.append({
        "time": datetime.datetime.now().isoformat(),
        "user": username,
        "question": user_msg,
        "answer": str(result),
        "reasoning": reasoning_output,
        "context": rag_context[:500],
    })
    return {
        "answer": result,
        "context_used": rag_context,
        "reasoning": reasoning_output,
        "geospatial_data_used": geospatial_result
    }

# MCP exposure for tools (addition, get_weather)
from fastapi_mcp import FastApiMCP
mcp = FastApiMCP(app)
mcp.mount()

@app.get("/addition")
def mcp_addition(a: int, b: int):
    return {"result": addition(a, b)}

@app.get("/weather")
def mcp_weather(city: str):
    return {"result": get_weather(city)}

@app.delete("/chat/reset")
def reset_history(username: str = Depends(verify_credentials)):
    memory = defaultdict(list)
    memory[username] = []
    return {"message": "Memory cleared."}

# To launch:
# uvicorn main:app --reload


@app.post("/chat")
async def chat(message: str, tool: str = None, args: dict = {}):
    if tool == "earth_engine_tool":
        result = earth_engine_tool(**args)
    elif tool == "climate_tool":
        result = climate_tool(**args)
    else:
        result = f"LLM response to: '{message}'"
    return {"result": result}

# MCP auto-exposition : tes endpoints sont documentés et accessibles
mcp = FastApiMCP(app)
mcp.mount()

# Optionnel : endpoints directs GET si tu veux test manuel
@app.get("/earth_engine_tool")
def mcp_earth_engine(lon: float, lat: float, date: str, radius: int = 10):
    return {"result": earth_engine_tool(lon, lat, date, radius)}

@app.get("/climate_tool")
def mcp_climate(location: str, info_type: str):
    return {"result": climate_tool(location, info_type)}

# Reset endpoint
@app.delete("/chat/reset")
def reset_history():
    return {"message": "Memory cleared."}
