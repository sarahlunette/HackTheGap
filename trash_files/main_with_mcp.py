"""
FASTAPI MVP - RAG + Reasoning Model (Mistral) + Claude Sonnet 4.5 + MCP Tools INTEGRATED
Now supporting direct MCP tool calls (no separate server): Earth Engine, Climate, OSM.
"""

import os
import csv
import re
import datetime
import logging
import json
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

import requests
from fastapi import Depends, APIRouter
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
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

# Memory
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# HuggingFace login
from huggingface_hub import login

import asyncio
import subprocess  # For running build_vectorstore.py

# Import MCP tools directly
import sys
sys.path.append('../mcp_server')
from tools.earth_engine_tool import fetch_earth_engine_data
from tools.climate_tool import run_climate_forecast_tool
from tools.osm_tool import run_osm_data_tool


# ============================================================
# ENV & CONFIG
# ============================================================
load_dotenv()

AUTH_MODE = os.getenv("AUTH_MODE", "basic")
MVP_USER = os.getenv("MVP_USER", "admin")
MVP_PASS = os.getenv("MVP_PASS", "password")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise RuntimeError("Missing CLAUDE_API_KEY")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "island_docs")

DOCS_DIR = Path("./docs")
DOCS_DIR.mkdir(exist_ok=True)

EXPORT_DIR = Path("./exports")
EXPORT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mvp")
# logger.setLevel(logging.DEBUG) # TODO:TMP

# Optional logs (for /logs endpoints if you want later)
ACTION_LOGS: list[Dict[str, Any]] = []

# ============================================================
# RAG INIT (Qdrant + LlamaIndex)
# ============================================================
logger.info("Initializing RAG with Qdrant vectorstore...")

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

USER_MEMORIES = defaultdict(
    lambda: ConversationBufferMemory(
        return_messages=True,
        chat_memory=ChatMessageHistory()
    )
)
# ============================================================
# RAG helper
# ============================================================
def query_knowledge_base(question: str) -> str:
    try:
        nodes = query_engine.retrieve(question)
    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        return ""
    return "\n".join(n.text for n in nodes)

# ============================================================
# DIRECT MCP TOOL CALLS (integrated)
# ============================================================

async def call_direct_earth_engine(lon, lat, date, radius):
    """
    Direct call to fetch_earth_engine_data tool.
    """
    try:
        result = await fetch_earth_engine_data(
            lon=float(lon),
            lat=float(lat),
            recent_start=str(date),
            radius=int(radius)
        )
        return result
    except Exception as e:
        return {"error": f"Earth Engine tool failed: {str(e)}"}

async def call_direct_climate(date_str, area):
    """
    Direct call to run_climate_forecast_tool.
    date_str: 'YYYY-MM'
    area: [N, W, S, E]
    """
    try:
        result = await run_climate_forecast_tool(date_str, area)
        return result
    except Exception as e:
        return {"error": f"Climate tool failed: {str(e)}"}

async def call_direct_osm(location, features):
    """
    Direct call to run_osm_data_tool.
    location: str
    features: list[str]
    """
    try:
        result = await run_osm_data_tool(location, features)
        return result
    except Exception as e:
        return {"error": f"OSM tool failed: {str(e)}"}

# ============================================================
# Reasoning Model Prompt (Mistral) - EXTENDED
# ============================================================
REASONING_PROMPT = """
You are a reasoning model responsible for extracting structured parameters from
the user's message so the crisis-resilience assistant can decide whether to:

A) generate a narrative/plan, OR
B) call one of the MCP tools: Earth Engine, Climate Forecast, OSM Data.

Your output MUST be a strictly valid JSON object with the following structure:

{
  "intent": "simple_question" | "resilience_plan" | "technical_analysis" | "geospatial_request" | "climate_request" | "osm_request",
  "entities": {
    "sectors": [...],
    "locations": [...],                   # e.g. ["Saint-Martin"]
    "time_horizon": "24h" | "72h" | "short_term" | "medium_term" | "long_term" | null,
    "specific_locations": [...],          # hospitals, ports, etc
    "disaster_type": string | null,       # e.g. "cyclone", "earthquake", "flood"
    "disaster_name": string | null,       # e.g. "Irma", "Maria"

    "date": "YYYY-MM-DD" | null,          # For Earth Engine
    "date_month": "YYYY-MM" | null,       # For Climate (monthly)
    "lon": float | null,                  # For Earth Engine
    "lat": float | null,                  # For Earth Engine
    "radius": float | null,               # For Earth Engine
    "area": list[float] | null,           # For Climate: [N, W, S, E]
    "osm_location": string | null,        # For OSM: location name
    "osm_features": list[string] | null   # For OSM: ["hospital", "airport", etc]
  },
  "response_mode": "short" | "structured"
}

---------------------------------------------------------------------------
### LOCATION AND DISASTER NAME EXTRACTION
- Extract any place name mentioned: ("Saint-Martin", "Barbuda", "Port-au-Prince").
- Extract any disaster name: ("Irma", "Maria", "Ida").
- Extract disaster type if obvious ("cyclone", "hurricane", "flood").

### DATE EXTRACTION RULES
- If the user explicitly writes a date ("2025-11-01", "1 Nov 2025"), extract and convert to ISO for "date".
- If the user mentions a well-known disaster name with a globally known date (e.g. "Cyclone Irma"), you may set the date.
- For climate forecasts, extract "date_month" as "YYYY-MM" if mentioned (e.g. "forecast for November 2025" → "2025-11").
- Never invent dates for generic phrases ("last week", "a while ago").

### GEO EXTRACTION RULES (STRICT)
- For Earth Engine: lon/lat only if explicitly provided in message (e.g. "lon 14.5, lat -22.1").
- For Climate: area as [N, W, S, E] if provided, else default to island bounds.
- For OSM: osm_location from place names, osm_features from keywords like "hospitals", "roads", "airports".

### MCP TRIGGER RULES
- "geospatial_request": ALL of date, lon, lat present.
- "climate_request": date_month present (and intent about forecast/weather).
- "osm_request": osm_location and osm_features present (and intent about infrastructure/data).
- If ANY required params missing → intent MUST NOT be the tool intent.

### INTENT CLASSIFICATION
- If lon+lat+date present → "geospatial_request".
- If asking about weather/forecast/climate → "climate_request".
- If asking for infrastructure data (hospitals, roads) → "osm_request".
- If the user asks about impacts, reconstruction, damage, analysis, planning: → "resilience_plan".
- If the user asks a factual question: → "simple_question".
- When unsure, choose the simplest faithful option: "simple_question".

### RADIUS EXTRACTION
- Extract only explicit numeric radius ("within 25 meters", "buffer of 500m").
- Remove units.
- If no radius mentioned → set radius = null.

### OUTPUT RULES
- Return ONLY valid JSON. No markdown, no comments, no explanation.
- Missing elements must be null — never invent coordinates or dataset names.

User message:
"{user_question}"
"""



def _default_structured_reasoning() -> dict:
    return {
        "intent": "resilience_plan",
        "response_mode": "structured",
        "entities": {
            "sectors": ["energy", "water"],
            "locations": [],
            "time_horizon": "short_term",
            "specific_locations": [],
            "disaster_type": None,
            "dataset": None,
            "date": None,
            "date_month": None,
            "lon": None,
            "lat": None,
            "radius": None,
            "area": None,
            "osm_location": None,
            "osm_features": None,
        },
    }

# -------------------------------
# Utility: Extract JSON cleanly
# -------------------------------
def extract_json_block(text: str) -> str:
    cleaned = text.replace("```json", "").replace("```", "").strip()

    brace_level = 0
    start = None

    for i, char in enumerate(cleaned):
        if char == "{":
            if brace_level == 0:
                start = i
            brace_level += 1

        elif char == "}":
            brace_level -= 1
            if brace_level == 0 and start is not None:
                return cleaned[start:i + 1].strip()

    raise ValueError(f"No JSON object found in text:\n{cleaned}")

# -------------------------------
# Call Retry
# -------------------------------
def mistral_call_with_retry(prompt, model="mistral-medium-latest", retries=5):
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
                time.sleep(wait)
                continue
            raise

    raise RuntimeError("Mistral: too many retries")

# ============================================================
# Reasoning function (FINAL) - EXTENDED
# ============================================================
def generate_reasoning_with_mistral(user_question: str) -> dict:
    if not MISTRAL_API_KEY:
        logger.warning("MISTRAL_API_KEY missing — fallback mode.")
        return _default_structured_reasoning()

    client = Mistral(api_key=MISTRAL_API_KEY)
    prompt = REASONING_PROMPT.replace("{user_question}", user_question)

    try:
        # ---------------------------
        # Call Mistral API
        # ---------------------------
        res = client.chat.complete(
            model="mistral-medium-latest",  # safest, always available
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000,
        )

        raw = res.choices[0].message.content.strip()
        logger.debug(f"[RAW OUTPUT]\n{raw}")

        # ---------------------------
        # Extract valid JSON
        # ---------------------------
        json_text = extract_json_block(raw)
        logger.debug(f"[EXTRACTED JSON]\n{json_text}")

        result = json.loads(json_text)

    except Exception as e:
        logger.error(f"[Reasoning] Mistral SDK error: {e}")
        return _default_structured_reasoning()

    # ------------------------------------------------------
    # Guarantee entity structure
    # ------------------------------------------------------
    result.setdefault("entities", {})
    entities = result["entities"]

    # ------------------------------------------------------
    # Normalize lon/lat/radius
    # ------------------------------------------------------
    for field in ["lon", "lat", "radius"]:
        value = entities.get(field)
        try:
            entities[field] = float(value) if value not in (None, "", "null") else None
        except Exception:
            entities[field] = None

    # ------------------------------------------------------
    # Normalize area for climate
    # ------------------------------------------------------
    area = entities.get("area")
    if area and isinstance(area, list) and len(area) == 4:
        entities["area"] = [float(x) for x in area]
    else:
        entities["area"] = None

    # ------------------------------------------------------
    # Normalize date → ISO format
    # ------------------------------------------------------
    raw_date = entities.get("date")
    if raw_date:
        parsed = None
        fmts = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d")
        for fmt in fmts:
            try:
                parsed = datetime.datetime.strptime(raw_date, fmt)
                break
            except Exception:
                continue

        entities["date"] = parsed.strftime("%Y-%m-%d") if parsed else None
    else:
        entities["date"] = None

    return result

# ============================================================
# Claude generator
# ============================================================
def generate_with_claude(prompt: str) -> str:
    with anthropic_client.messages.stream(
        model=CLAUDE_MODEL,
        temperature=0.7,
        max_tokens=64000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        return stream.get_final_text()


# ============================================================
# Auth & FastAPI
# ============================================================
app = FastAPI(title="Crisis RAG + MCP Integrated API")
security = HTTPBasic()


def verify_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
    authorization: str = Header(None),
):
    if AUTH_MODE == "google":
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing Bearer token")
        # For brevity we don't validate with Google here.
        return authorization.split(" ")[1]

    if credentials.username != MVP_USER or credentials.password != MVP_PASS:
        raise HTTPException(401, "Unauthorized")

    return credentials.username


# ============================================================
# Chat endpoint - WITH DIRECT MCP CALLS
# ============================================================
class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: ChatRequest, username: str = Depends(verify_credentials)):
    memory = USER_MEMORIES[username]
    user_msg = req.question.strip()

    # 1. Reasoning
    reasoning_output = generate_reasoning_with_mistral(user_msg)
    entities = reasoning_output.get("entities", {})
    print(entities)

    # 2. Initial RAG
    rag_context = query_knowledge_base(user_msg)
    rag_block = rag_context if rag_context.strip() else "<<EMPTY>>"

    # 3. History (last 5 messages)
    history = "\n".join(
        f"{getattr(m, 'type', 'UNKNOWN').upper()}: {getattr(m, 'content', '')}"
        for m in memory.chat_memory.messages[-5:]
    )

    # 4. Reasoning metadata
    reasoning_metadata = f"""
### 🔍 REASONING MODEL ANALYSIS (INTERNAL)
Intent: {reasoning_output.get('intent')}
Response Mode: {reasoning_output.get('response_mode')}
Entities: {json.dumps(entities, ensure_ascii=False)}
"""

    # 5. Optional DIRECT MCP Tool calls
    tool_results = {}

    # Earth Engine
    if (
        reasoning_output.get("intent") == "geospatial_request"
        and entities.get("lon") is not None
        and entities.get("lat") is not None
        and entities.get("date") is not None
    ):
        try:
            tool_results["earth_engine"] = await call_direct_earth_engine(
                lon=entities["lon"],
                lat=entities["lat"],
                date=entities["date"],
                radius=int(entities.get("radius") or 10)
            )
        except Exception as e:
            logger.error(f"Direct Earth Engine failed: {e}")
            tool_results["earth_engine"] = {"error": f"Direct Earth Engine failed: {e}"}

    # Climate
    elif (
        reasoning_output.get("intent") == "climate_request"
        and entities.get("date_month") is not None
    ):
        area = entities.get("area") or [18.2, -63.2, 18.0, -62.9]  # Default Saint-Martin
        try:
            tool_results["climate"] = await call_direct_climate(
                date_str=entities["date_month"],
                area=area
            )
        except Exception as e:
            logger.error(f"Direct Climate failed: {e}")
            tool_results["climate"] = {"error": f"Direct Climate failed: {e}"}

    # OSM
    elif (
        reasoning_output.get("intent") == "osm_request"
        and entities.get("osm_location") is not None
        and entities.get("osm_features") is not None
    ):
        try:
            tool_results["osm"] = await call_direct_osm(
                location=entities["osm_location"],
                features=entities["osm_features"]
            )
        except Exception as e:
            logger.error(f"Direct OSM failed: {e}")
            tool_results["osm"] = {"error": f"Direct OSM failed: {e}"}

    # If any tool was called, refresh RAG
    if tool_results:
        # Trigger vectorstore rebuild (sync)
        try:
            subprocess.run(["python", "build_vectorstore.py"], check=True)
            logger.info("Vectorstore rebuilt after tool call.")
        except Exception as e:
            logger.error(f"Failed to rebuild vectorstore: {e}")

        # Refresh RAG context
        rag_context = query_knowledge_base(user_msg)
        rag_block = rag_context if rag_context.strip() else "<<EMPTY>>"

    # 6. Build full Claude prompt
    prompt = f"""
{reasoning_metadata}
-------------------------------------------------------------------------------
### 🔎 INPUT BLOCKS

1. **Reasoning Model Output (summarized above)** — structured guidance about the user’s intent, sectors, locations, and time horizon.
2. **RAG CONTEXT** — text retrieved from local documents (GIS, infrastructure, reports, tables, project docs).
3. **CONVERSATION HISTORY** — the last turns of the chat with this user.
4. **CURRENT USER MESSAGE** — the question to answer now.
5. **TOOL RESULTS** — any data fetched from MCP tools (Earth Engine, Climate, OSM).

---

#### RAG CONTEXT
<<<
{rag_block}
>>>

#### CONVERSATION HISTORY
<<<
{history}
>>>

#### TOOL RESULTS
<<<
{json.dumps(tool_results, ensure_ascii=False, indent=2) if tool_results else "No tools called."}
>>>

#### CURRENT USER MESSAGE
<<<
{user_msg}
>>>

-------------------------------------------------------------------------------
### 🎯 GLOBAL ROLE

You are **RESILIENCE-GPT**, a Crisis & Resilience Strategic Planner AI for small islands, coastal territories, and fragile states. You specialize in:

- Post-disaster damage assessment and impact mapping
- Multi-sector resilience engineering and infrastructure recovery
- Critical infrastructure prioritization (power, water, health, telecom, transport)
- Humanitarian logistics and supply-chain restoration
- GIS-informed planning and geospatial reasoning (elevation, exposure, chokepoints)
- Climate risk modelling and long-term adaptation
- Economic and financial reconstruction strategies
- Long-term resilience transformation planning (1–15 years)

You must integrate relevant information from the RAG CONTEXT and TOOL RESULTS when available.

-------------------------------------------------------------------------------
### 🧠 MODE SELECTION (SHORT vs STRUCTURED)

The Reasoning Model suggests:
- **Intent** = {reasoning_output.get('intent')}
- **Response Mode** = {reasoning_output.get('response_mode')}

Behavior:

1. If `response_mode = "short"` and the user is asking a simple, factual, or conceptual question:
   - Answer in 1–3 short paragraphs, conversational and clear.

2. If `response_mode = "structured"` or the user explicitly asks for a plan / strategy / roadmap / prioritization:
   - Produce a multi-section, highly detailed resilience plan.
   - Focus on prioritization and project-level detail.

You must not ask the user for clarification; choose the best interpretation and answer directly.

-------------------------------------------------------------------------------
### 🧭 RAG INTEGRATION & GAP HANDLING

- If RAG CONTEXT is non-empty: extract concrete facts and use them.
- If TOOL RESULTS are present: incorporate the fetched data into your analysis.
- If RAG CONTEXT is `<<EMPTY>>`: rely on best practices for similar territories.
- Explicitly state when you rely on generic assumptions.

-------------------------------------------------------------------------------
### 📘 STRUCTURED OUTPUT FORMAT (ONLY IF STRUCTURED MODE)

[... keep your detailed section structure here if you want ...]
(Executive Summary, Context Reconstruction, Priority Matrix, Sector Plans, Project Portfolio, Logistics, Finance, Risks, Roadmap.)

In short mode, answer briefly without the full structure.

Now answer the CURRENT USER MESSAGE accordingly.
"""

    # 7. Prompt length safety
    MAX_PROMPT_CHARS = 600_000
    safe_prompt = prompt[:MAX_PROMPT_CHARS]

    # 8. Ask Claude
    answer = generate_with_claude(safe_prompt)

    # 9. Memory update & logs
    memory.chat_memory.add_user_message(user_msg)
    memory.chat_memory.add_ai_message(answer)

    ACTION_LOGS.append(
        {
            "time": datetime.datetime.now().isoformat(),
            "user": username,
            "question": user_msg,
            "answer": answer,
            "context": rag_context[:500],
            "reasoning": reasoning_output,
            "tool_results": tool_results,
        }
    )

    # 10. Response
    return {
        "answer": answer,
        "context_used": rag_context,
        "reasoning": reasoning_output,
        "extracted_date": entities.get("date"),
        "extracted_lon": entities.get("lon"),
        "extracted_lat": entities.get("lat"),
        "extracted_radius": entities.get("radius"),
        "tool_results": tool_results,
        "conversation_turns": len(memory.chat_memory.messages) // 2,
    }


# ============================================================
# Reset history
# ============================================================
@app.delete("/chat/reset")
def reset_history(username: str = Depends(verify_credentials)):
    USER_MEMORIES[username] = ConversationBufferMemory(
        return_messages=True,
        chat_memory=ChatMessageHistory()
    )
    return {"message": "Memory cleared."}


# Test feature
# ======================================================================

if __name__ == "__main__":
    import asyncio

    print("=== TESTING /chat FUNCTION DIRECTLY ===")

    async def test_chat():
        class FakeReq:
            question = "Resilience plan of Cyclone Irma on Saint-Martin? On what date? At what latitude of the center of the island? At what longitude of the center of the island? Make sure structured answer with latitude, longitude and radius = 10, find the date."

        out = await chat(FakeReq(), "admin")
        print(json.dumps(out, indent=2, ensure_ascii=False))

    asyncio.run(test_chat())
