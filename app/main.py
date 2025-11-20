"""
FASTAPI MVP — RAG + Reasoning Model (Mistral) + Claude Sonnet 4.5 + MCP Tools
Now supporting async MCP Earth Engine data ingestion directly from /chat.
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
from fastapi.middleware.cors import CORSMiddleware

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
from fastmcp import Client

MCP_CLIENT_URL = "http://mcp_server:9001/mcp"


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
# CALL MCP EARTH ENGINE
# ============================================================

async def call_mcp_fetch_earth_engine(lon, lat, date, radius):
    """
    Calls MCP server tool 'fetch_earth_engine_data' using FastMCP Client API.
    """

    payload = {
        "lon": float(lon),
        "lat": float(lat),
        "recent_start": str(date),
        "radius": int(radius)
    }

    try:
        client = Client(MCP_CLIENT_URL)

        async with client:  # ensures connection open/close
            result = await client.call_tool("fetch_earth_engine_data", payload)
            return result

    except Exception as e:
        return {"error": f"MCP client call failed: {str(e)}"}


# ============================================================
# Reasoning Model Prompt (Mistral)
# ============================================================
REASONING_PROMPT = """
You are a reasoning model responsible for extracting structured parameters from
the user's message so the crisis-resilience assistant can decide whether to:

A) generate a narrative/plan, OR  
B) call the geospatial MCP tool fetch_earth_engine_data.

Your output MUST be a strictly valid JSON object with the following structure:

{
  "intent": "simple_question" | "resilience_plan" | "technical_analysis" | "geospatial_request",
  "entities": {
    "sectors": [...],
    "locations": [...],                   # e.g. ["Saint-Martin"]
    "time_horizon": "24h" | "72h" | "short_term" | "medium_term" | "long_term" | null,
    "specific_locations": [...],          # hospitals, ports, etc
    "disaster_type": string | null,       # e.g. "cyclone", "earthquake", "flood"
    "disaster_name": string | null,       # e.g. "Irma", "Maria"

    "date": "YYYY-MM-DD" | null,          # If explicitly provided or clearly inferable
    "lon": float | null,                  # ONLY if explicitly provided in message
    "lat": float | null,                  # ONLY if explicitly provided in message
    "radius": float | null                # only explicit numeric values (“within 30m”)
  },
  "response_mode": "short" | "structured"
}

---------------------------------------------------------------------------
### LOCATION AND DISASTER NAME EXTRACTION
- Extract any place name mentioned: ("Saint-Martin", "Barbuda", "Port-au-Prince").
- Extract any disaster name: (“Irma”, “Maria”, “Ida”).
- Extract disaster type if obvious (“cyclone”, “hurricane”, “flood”).
- Infer coordinates from locations. Only set lon/lat if you can find them.

### DATE EXTRACTION RULES
- If the user explicitly writes a date (“2025-11-01”, “1 Nov 2025”), extract and convert to ISO.
- If the user mentions a well-known disaster name with a **globally known date** (e.g. “Cyclone Irma”),
  you may set the date.
    Example:
      “Hurricane Irma” → date = “2017-09-06”.
- Never invent dates for generic phrases ("last week", "a while ago").

### GEO EXTRACTION RULES (STRICT)
- Example accepted: “lon 14.5, lat -22.1”
- Example rejected: “the north of the island” → lon = null, lat = null

### MCP TRIGGER RULES (VERY STRICT)
#The MCP tool should only be triggered if ALL of the following are explicitly present:
1. a valid ISO date
2. longitude (number)
3. latitude (number)

If ANY of these are missing → intent MUST NOT be “geospatial_request”.

### INTENT CLASSIFICATION
- If lon+lat+date+dataset all present → intent = “geospatial_request”.
- If the user asks about impacts, reconstruction, damage, analysis, planning:
      → intent = “resilience_plan”.
- If the user asks a factual question:
      → intent = “simple_question”.
- When unsure, choose the simplest faithful option: “simple_question”.

### RADIUS EXTRACTION
- Extract only explicit numeric radius (“within 25 meters”, “buffer of 500m”).
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
            "lon": None,
            "lat": None,
            "radius": None,
        },
    }

# -------------------------------
# Utility: Extract JSON cleanly
# -------------------------------
def extract_json_block(text: str) -> str:
    cleaned = text.replace("```json", "").replace("```", "")

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
                time.sleep(wait)
                continue
            raise

    raise RuntimeError("Mistral: too many retries")

# ============================================================
# Reasoning function (FINAL)
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
    # Normalize user date → ISO format
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
app = FastAPI(title="Crisis RAG + MCP API")
security = HTTPBasic()

# Autoriser toutes les origines (comme votre CORS(app) en Flask)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod, remplace par la liste de domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Chat endpoint
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

    # 5. Optional MCP Tool call (geospatial)
    geospatial_result = None

    if (
        reasoning_output.get("intent") == "geospatial_request"
        and entities.get("lon") is not None
        and entities.get("lat") is not None
        and entities.get("date") is not None
    ):
        try:
            geospatial_result = await call_mcp_fetch_earth_engine(
                lon=entities["lon"],
                lat=entities["lat"],
                date=entities["date"],
                radius=int(entities.get("radius") or 10)
            )

        except Exception as e:
            logger.error(f"MCP tool failed: {e}")
            geospatial_result = {"error": f"MCP tool failed: {e}"}

        # After MCP tool updates docs + vectorstore, refresh RAG
        rag_context = query_knowledge_base(user_msg)
        rag_block = rag_context if rag_context.strip() else "<<EMPTY>>"

    # 6. Build full Claude prompt
    prompt = f"""
{reasoning_metadata}
-------------------------------------------------------------------------------
### 🔎 INPUT BLOCKS

You receive four inputs:

1. **Reasoning Model Output (summarized above)** — structured guidance about the user’s intent, sectors, locations, and time horizon.
2. **RAG CONTEXT** — text retrieved from local documents (GIS, infrastructure, reports, tables, project docs).
3. **CONVERSATION HISTORY** — the last turns of the chat with this user.
4. **CURRENT USER MESSAGE** — the question to answer now.

---

#### RAG CONTEXT
<<<
{rag_block}
>>>

#### CONVERSATION HISTORY
<<<
{history}
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

You must integrate relevant information from the RAG CONTEXT when available.

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
        "geospatial_data_used": geospatial_result,
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
            question =  "Resilience plan of Cyclone Irma on Saint-Martin? On what date ? At what latitude of the center of the island ? At what longitude of the center of the island ? Make sure structured answer with latitude, longitude and radius = 10, find the date."


        out = await chat(FakeReq(), "admin")
        print(json.dumps(out, indent=2, ensure_ascii=False))

    asyncio.run(test_chat())
