import os
import re
import logging
import json
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from collections import defaultdict
import datetime

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# LangChain/Claude agent support
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_community.tools import Tool
# from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

from langgraph.graph import StateGraph, END
from langchain_core.tools import Tool

from fastapi_mcp import FastApiMCP

from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI

from tools.earth_engine_tool import fetch_earth_engine_data
from tools.climate_tool import get_climate_forecast
from tools.osm_tool import get_osm_data

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mvp")

app = FastAPI(title="LangChain Claude + MCPAgent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH_MODE = os.getenv("AUTH_MODE", "basic")
MVP_USER = os.getenv("MVP_USER", "admin")
MVP_PASS = os.getenv("MVP_PASS", "password")
security = HTTPBasic()

# --- RAG Context ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "island_docs")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant_client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, storage_context=storage_context, embed_model=embed_model,
)
query_engine = index.as_retriever(similarity_top_k=3)


def query_knowledge_base(question: str) -> str:
    try:
        nodes = query_engine.retrieve(question)
    except Exception as e:
        logging.error(f"Error querying vector store: {e}")
        return ""
    return "\n".join(n.text for n in nodes)

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != MVP_USER or credentials.password != MVP_PASS:
        raise HTTPException(401, "Unauthorized")
    return credentials.username

class ChatRequest(BaseModel):
    question: str

# --- MCP Agent Setup ---
CONFIG = {
    "mcpServers": {
        "public-demo-fastapi": {"url": "http://localhost:9001/mcp"}
    }
}

# Mistral Reasoning Model Explicit Agent Setup
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("Missing MISTRAL_API_KEY")
mistral_llm = ChatMistralAI(
    model="mistral-medium-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# Claude Synthesis Model Explicit Agent Setup
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise RuntimeError("Missing CLAUDE_API_KEY")
claude_llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    api_key=os.getenv("CLAUDE_API_KEY")
)

# 2. Make a list of tools for LangChain agent
tools = [get_climate_forecast, fetch_earth_engine_data, get_osm_data]

# 3. Create memory (per user is best; for demo, global)

# 4. Initialize LangChain agent with Claude and tools
REASONING_PROMPT = """
You are a reasoning model for a crisis-resilience assistant. 
Your responsibilities:
A) Generate a narrative or plan when needed,
B) Call the geospatial MCP tool fetch_earth_engine_data or get_climate_data *only* when sufficient parameters are present.

Follow these steps strictly:

1. Extract these parameters from the user's message, ONLY if explicitly present:
   - date (ISO format "YYYY-MM-DD"),
   - latitude (as a number),
   - longitude (as a number).

2. If—and only if—all three (date, lat, lon) are present:
   - Set intent to "geospatial_request".
   - Imagine calling the MCP tool fetch_earth_engine_data using those parameters and synthesize a plausible response (do not execute any real call).
   - All used parameters must be returned in the JSON.

3. Otherwise, perform extraction/classification:
   - Extract: intent, entities (sectors, locations, specific_locations, disaster_type, disaster_name, radius, etc.).
   - intent must be "resilience_plan", "simple_question", or "technical_analysis" based on the user's message.

4. ALWAYS return one strictly valid JSON object (no markdown, no comments, no explanations). 
   - Every missing element must be set as null. 
   - Never invent coordinates or dataset names.
   - Example output:
{
  "intent": "geospatial_request" | "resilience_plan" | "technical_analysis" | "simple_question",
  "entities": {
    "sectors": [...],
    "locations": [...],                     # e.g. ["Saint-Martin"]
    "time_horizon": "24h" | "72h" | "short_term" | "medium_term" | "long_term" | null,
    "specific_locations": [...],            # hospitals, ports, etc.
    "disaster_type": string | null,         # e.g. "cyclone", "earthquake", "flood"
    "disaster_name": string | null,         # e.g. "Irma", "Maria"
    "date": "YYYY-MM-DD" | null,            # only if explicitly provided
    "lon": float | null,                    # only if explicitly provided
    "lat": float | null,                    # only if explicitly provided
    "radius": float | null                  # only if explicitly numeric ("within 30m", "buffer 500m")
  },
  "response_mode": "short" | "structured"
}

---------------------------------------------------------------------------
### LOCATION AND DISASTER NAME EXTRACTION
- Extract all place names (e.g. "Saint-Martin", "Barbuda", "Port-au-Prince").
- Extract all disaster names (e.g. "Irma", "Maria", "Ida").
- Extract disaster type if obvious ("cyclone", "hurricane", "flood").
- Infer coordinates from locations ONLY if explicitly given. Do not guess.

### DATE EXTRACTION RULES
- Extract dates only if written as "YYYY-MM-DD" or similar; convert to ISO.
- If a well-known disaster is named ("Cyclone Irma"), you may set its globally known date.
- Do not invent dates for vague phrases ("last week", "recently").

### GEO EXTRACTION RULES (STRICT)
- Accept: explicit numeric statements like "lon 14.5, lat -22.1".
- Reject and set to null: vague descriptions ("north of the island").

### MCP TRIGGER RULES (VERY STRICT)
- The MCP tool should only be triggered if ALL of:
    1. valid ISO date,
    2. longitude (number),
    3. latitude (number).
- If ANY of these are missing, intent MUST NOT be "geospatial_request".

### INTENT CLASSIFICATION
- If lon + lat + date + dataset present: intent = "geospatial_request".
- If discussing impacts, planning, reconstruction, analysis: intent = "resilience_plan".
- If asking facts: intent = "simple_question".
- When in doubt, choose the simplest faithful option: "simple_question".

### RADIUS EXTRACTION
- Only extract explicit numeric radius ("within 25 meters", "buffer 500m").
- Remove units.
- If no explicit radius, set to null.

### OUTPUT RULES
- Only output valid JSON as shown above. No markdown, comments, or explanations.
- All missing elements must be null. Never invent coordinates or dataset names.
"""

mistral_agent = create_agent(model=mistral_llm, tools=tools, system_prompt=REASONING_PROMPT)
claude_agent = create_agent(model=claude_llm, tools=tools)

# MCP mounting (for interactive /mcp tools if desired)
mcp = FastApiMCP(app)
mcp.mount()

def clean_llm_json(text):
    # Remove: starting ''', optional "json" with spaces, and ending '''
    cleaned = re.sub(
        r"^'''\s*json\s*|'''$",  # match '''json (start) or ''' (end)
        "",
        text.strip(),
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()
    return cleaned

# ---------- Graph: Mistral -> Tools -> Claude ----------
@app.post("/agent/mistral")
async def use_agent_mistral(req: ChatRequest, username: str = Depends(verify_credentials)):
    prompt = req.question.strip()
    result = mistral_agent.invoke({"messages": [{"role": "user", "content": prompt }]})
    return result

@app.post("/chat/mistral-claude")
async def chat(req: ChatRequest, username: str = Depends(verify_credentials)):
    # memory = USER_MEMORIES[username]
    user_msg = req.question.strip()

    # 1. Reasoning
    reasoning_output = mistral_agent.invoke({"messages": [{"role": "user", "content": REASONING_PROMPT + user_msg }]})
    entities = clean_llm_json(reasoning_output).get("entities", {})
    print(entities)

    # 4. Reasoning metadata
    reasoning_metadata = f"""
### 🔍 REASONING MODEL ANALYSIS (INTERNAL)
Intent: {reasoning_output.get('intent')}
Response Mode: {reasoning_output.get('response_mode')}
Entities: {json.dumps(entities, ensure_ascii=False)}
"""

    # RAG context # TODO refresh rag after use of agent ? The agent refreshes the rag itself ?
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
    answer = claude_llm.invoke(safe_prompt)

    # 10. Response
    return {
        "answer": answer,
        "context_used": rag_context,
        "reasoning": reasoning_output,
        "extracted_date": entities.get("date"),
        "extracted_lon": entities.get("lon"),
        "extracted_lat": entities.get("lat"),
        "extracted_radius": entities.get("radius")
    }

