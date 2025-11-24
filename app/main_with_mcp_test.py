# TODO: Comment code

# Imports
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

# Vectorstore
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

# FastAPI MCP
from fastapi_mcp import FastApiMCP

# LLM
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI

# Agent graph
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import Tool
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langgraph.graph import StateGraph, END

# Tools
from tools.earth_engine_tool import fetch_earth_engine_data
from tools.climate_tool import get_climate_forecast
from tools.osm_tool import get_osm_data
from langchain_community.tools import Tool

tools = [
    Tool(
        name="fetch_earth_engine_data",
        func=fetch_earth_engine_data,
        description="Fetch geospatial Earth Engine data for a given lat, lon, and date"
    ),
    Tool(
        name="get_climate_forecast",
        func=get_climate_forecast,
        description="Return climate forecast data for given location and date"
    ),
    Tool(
        name="get_osm_data",
        func=get_osm_data,
        description="Retrieve OpenStreetMap data for a given area"
    )
]


# ---- Add this near other imports ----
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langgraph")

# --- FastAPI ---
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

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != MVP_USER or credentials.password != MVP_PASS:
        raise HTTPException(401, "Unauthorized")
    return credentials.username

class ChatRequest(BaseModel):
    question: str

# MCP mounting (for interactive /mcp tools if desired)
mcp = FastApiMCP(app)
mcp.mount()

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

def initial_state():
    return {
        "user_msg": None,
        "reasoning": None,
        "tool_result": None,
        "final_answer": None
    }

#  Initialize LangChain agent with Claude and tools
REASONING_PROMPT = """
You are a reasoning model for a crisis-resilience assistant. 
Your responsibilities:
A) Generate a narrative or plan when needed,
B) Call the geospatial MCP tool fetch_earth_engine_data or get_climate_data *only* when you can infer sufficient parameters.

Follow these steps strictly:

1. Extract these parameters from the user's message, ONLY if explicitly present:
   - date (ISO format "YYYY-MM-DD"),
   - latitude (as a number),
   - longitude (as a number).

2. If—and all three (date, lat, lon) are present (otherwise infer them if possible):
   - Set intent to "geospatial_request".
   - Imagine calling the MCP tool fetch_earth_engine_data using those parameters and synthesize a plausible response (do not execute any real call).
   - All used parameters must be returned in the JSON.

3. Otherwise, perform extraction/classification:
   - Extract: intent, entities (sectors, locations, specific_locations, disaster_type, disaster_name, radius, etc.).
   - intent must be "resilience_plan", "simple_question", or "technical_analysis" based on the user's message. Only if you cannot infer the parameters for the tools do you not use them.
   - Still imagine calling the MCP tool fetch_earth_engine_data using those parameters and synthesize a plausible response (do not execute any real call).

4. ALWAYS return one strictly valid JSON object (no markdown, no comments, no explanations). 
   - Every missing element must be set as null. 
   - Never invent coordinates or dataset names. Try to infer them though.
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
- Infer coordinates from locations ONLY if explicitly given or that the location of the disaster is well-known.

### DATE EXTRACTION RULES
- Extract dates only if written as "YYYY-MM-DD" or similar; convert to ISO.
- If a well-known disaster is named ("Cyclone Irma"), you may set its globally known date.
- Do not invent dates for vague phrases ("last week", "recently").

### GEO EXTRACTION RULES (STRICT)
- Accept: explicit numeric statements like "lon 14.5, lat -22.1".
- Reject and set to null: vague descriptions ("north of the island").

### MCP TRIGGER RULES (VERY STRICT)
- The MCP tool should only be triggered if the LLM can infer ALL of:
    1. valid ISO date,
    2. longitude (number),
    3. latitude (number).

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

graph = StateGraph(initial_state)

class ToolDebugCallback(BaseCallbackHandler):
    def on_tool_start(self, tool, input, **kwargs):
        print(f"[TOOL START] {tool.name} called with input: {input}")

    def on_tool_end(self, output, **kwargs):
        print(f"[TOOL END] Output: {output}")

# --- Helper Functions ---
def clean_llm_json(text):
    # Remove: starting ''', optional "json" with spaces, and ending '''
    cleaned = re.sub(
        r"^'''\s*json\s*|'''$",  # match '''json (start) or ''' (end)
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()
    return cleaned

# --- Reasoning LLM Node ---
def reasoning_fn(state):
    user_msg = state["user_msg"]
    prompt = REASONING_PROMPT + f"""
{user_msg}
"""
    reasoning_output = mistral_llm.invoke(prompt)
    try:
        reasoning_json = json.loads(clean_llm_json(reasoning_output))
    except Exception as e:
        logger.error(f"Reasoning JSON parse error: {e}")
        reasoning_json = {"entities": {"lat": None, "lon": None, "date": None}, "intent": None}
    return reasoning_json

graph.add_node('reasoning', reasoning_fn)

# --- Tool Node ---
def tool_earth_engine_fn(state):
    reasoning = state["reasoning"]
    lat = reasoning['entities'].get("lat")
    lon = reasoning['entities'].get("lon")
    date = reasoning['entities'].get("date")

    # If parameters complete, call the real tool
    if lat is not None and lon is not None and date is not None:
        result = fetch_earth_engine_data(lat=lat, lon=lon, date=date)
    else:
        result = "Tool not called — insufficient parameters."
    return {"tool_result": result}

graph.add_node("tool", tool_earth_engine_fn)

# TODO: Climate data (the output has to be a 4 digit direction for area transform with llm or after llm)

# --- Synthesis Node ---
def synthesis_fn(state):
    reasoning = state["reasoning"]
    tool_result = state["tool_result"]
    user_msg = state['user_msg']

    # Get RAG context
    rag_context = query_knowledge_base(user_msg)
    rag_block = rag_context if rag_context.strip() else "<<EMPTY>>"
    
    # 7. Prompt length safety
    synthesis_prompt = f"""
Reasoning info: {json.dumps(reasoning, ensure_ascii=False)}
Tool result: {tool_result}
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
- **Intent** = {json.dumps(reasoning, ensure_ascii=False).get('intent')}
- **Response Mode** = {json.dumps(reasoning, ensure_ascii=False).get('response_mode')}

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
    MAX_PROMPT_CHARS = 600_000
    safe_prompt = synthesis_prompt[:MAX_PROMPT_CHARS]

    final_answer = claude_llm.invoke(safe_prompt)
    return final_answer

graph.add_node("synthesis", synthesis_fn)

# --- Build the graph ---
graph.add_edge(START, "reasoning")
graph.add_edge("reasoning", "tool")
graph.add_edge("tool", "synthesis")
graph.add_edge("synthesis", END)
graph_app = graph.compile()

# --- Run the graph ---
def run_resilience_pipeline(user_msg: str):
    outputs = graph_app.invoke({"user_msg": user_msg})
    return outputs

# ---------- Graph: Mistral -> Tools -> Claude ----------
@app.post("/agent/mistral")
async def use_agent_mistral(req: ChatRequest, username: str = Depends(verify_credentials)):
    user_msg = req.question.strip()
    
    # Step 1: Run Mistral agent for reasoning & parameter extraction
    reasoning_output_str = mistral_agent.invoke({"messages": [{"role": "user", "content": user_msg}]})
    
    # Step 2: Clean and parse LLM JSON output
    def clean_llm_json(text):
        cleaned = re.sub(
            r"^'''\s*json\s*|'''$", "", text, flags=re.IGNORECASE | re.MULTILINE
        ).strip()
        return cleaned

    try:
        reasoning_json = json.loads(clean_llm_json(reasoning_output_str))
    except Exception as e:
        logger.error(f"Reasoning JSON parse error: {e}")
        reasoning_json = {"entities": {"lat": None, "lon": None, "date": None}, "intent": None}

    entities = reasoning_json.get("entities", {})
    lat = entities.get("lat")
    lon = entities.get("lon")
    date = entities.get("date")

    # Step 3: If all required params are present, call the real tool
    tool_result = None
    if lat is not None and lon is not None and date is not None:
        tool_result = fetch_earth_engine_data(lat=lat, lon=lon, date=date)
    else:
        tool_result = None

    # Step 4: Build response: use tool result if available, else fallback to LLM's structured output
    return {
        "reasoning": reasoning_json,
        "tool_result": tool_result,
        "used_tool": bool(tool_result is not None)
    }

# TODO: changer pour une réponse intermédiaire du graph

@app.post("/chat/mistral-claude")
async def chat(user_req: ChatRequest, username: str = Depends(verify_credentials)):
    user_msg = user_req.question.strip()
    
    # Run the full graph pipeline
    outputs = run_resilience_pipeline(user_msg)

    # Get intermediary answers
    reasoning = outputs.reasoning
    
    # Get Claude final answer
    final_answer = outputs.final_answer

    # Response
    return {
        "answer": final_answer,
        # "context_used": rag_context, use rag context out in a node
        "reasoning": reasoning,
        "extracted_date": reasoning.entities.get("date"),
        "extracted_lon": reasoning.entities.get("lon"),
        "extracted_lat": reasoning.entities.get("lat"),
        "extracted_radius": reasoning.entities.get("radius")
    }

