###############################################
# agent.py — Unified LangGraph Agent
###############################################

import os
import json
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# ================
# Dependencies
# ================
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

# Mistral + Claude
from mistralai import Mistral
from anthropic import Anthropic

# RAG
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# MCP Client (FastMCP)
from fastmcp import Client

# ============================================
# Load ENV
# ============================================
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "island_docs")
MCP_CLIENT_URL  = os.getenv("MCP_CLIENT_URL", "http://mcp_server:9001/mcp")

anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

###############################################
# LOGGING
###############################################
logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

###############################################
# MEMORY PER USER
###############################################
USER_MEMORIES = {}

def get_memory(username: str):
    if username not in USER_MEMORIES:
        USER_MEMORIES[username] = ConversationBufferMemory(
            return_messages=True,
            chat_memory=ChatMessageHistory()
        )
    return USER_MEMORIES[username]

###############################################
# RAG INIT
###############################################
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
    embed_model=embed_model
)
query_engine = index.as_retriever(similarity_top_k=3)

def query_rag(question: str) -> str:
    try:
        nodes = query_engine.retrieve(question)
        return "\n".join(n.text for n in nodes)
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return ""

###############################################
# REASONING — MISTRAL
###############################################

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

def extract_json_block(text: str) -> str:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    brace = 0
    start = None
    for i, c in enumerate(cleaned):
        if c == "{":
            if brace == 0: start = i
            brace += 1
        elif c == "}":
            brace -= 1
            if brace == 0 and start is not None:
                return cleaned[start:i+1].strip()
    raise ValueError("No JSON found.")

def run_reasoning_mistral(user_question: str) -> dict:
    client = Mistral(api_key=MISTRAL_API_KEY)
    prompt = REASONING_PROMPT.replace("{user_question}", user_question)

    res = client.chat.complete(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )

    raw = res.choices[0].message.content.strip()
    json_text = extract_json_block(raw)
    parsed = json.loads(json_text)

    # normalize
    entities = parsed.setdefault("entities", {})
    for f in ["lon", "lat", "radius"]:
        try:
            entities[f] = float(entities.get(f)) if entities.get(f) else None
        except:
            entities[f] = None

    # format date → ISO
    if entities.get("date"):
        try:
            dt = datetime.datetime.fromisoformat(entities["date"])
            entities["date"] = dt.strftime("%Y-%m-%d")
        except:
            entities["date"] = None

    return parsed

###############################################
# MCP CALL
###############################################

async def call_mcp(entities: dict):
    try:
        client = Client(MCP_CLIENT_URL)
        payload = {
            "lon": entities["lon"],
            "lat": entities["lat"],
            "recent_start": entities["date"],
            "radius": int(entities.get("radius") or 10)
        }
        async with client:
            return await client.call_tool("fetch_earth_engine_data", payload)
    except Exception as e:
        return {"error": f"MCP failed: {e}"}

###############################################
# FINAL ANSWER — CLAUDE
###############################################
def build_claude_prompt(user_msg, reasoning, rag, mcp):
    return f"""
### REASONING MODEL
{json.dumps(reasoning, ensure_ascii=False, indent=2)}

### RAG CONTEXT
{rag or '<<EMPTY>>'}

### MCP RESULT
{json.dumps(mcp, ensure_ascii=False, indent=2) if mcp else 'None'}

### USER MESSAGE
{user_msg}

### ROLE
You are RESILIENCE-GPT...

-------------------------------------------------------------------------------
### 🔎 INPUT BLOCKS

You receive four inputs:

1. **Reasoning Model Output (summarized above)** — structured guidance about the user’s intent, sectors, locations, and time horizon.
2. **RAG CONTEXT** — text retrieved from local documents (GIS, infrastructure, reports, tables, project docs).
3. **CONVERSATION HISTORY** — the last turns of the chat with this user.
4. **CURRENT USER MESSAGE** — the question to answer now.

---

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

def run_claude(prompt: str) -> str:
    with anthropic_client.messages.stream(
        model="claude-sonnet-4.5",
        temperature=0.7,
        max_tokens=64000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        return stream.get_final_text()


###############################################
# LANGGRAPH — STATE
###############################################

class AgentState(BaseModel):
    messages: List[Dict[str, Any]]
    reasoning: Optional[Dict[str, Any]] = None
    rag: Optional[str] = None
    mcp: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None


###############################################
# NODES
###############################################

def reasoning_node(state: AgentState):
    user_msg = state.messages[-1]["content"]
    reasoning = run_reasoning_mistral(user_msg)
    return {"reasoning": reasoning}

def rag_node(state: AgentState):
    user_msg = state.messages[-1]["content"]
    rag = query_rag(user_msg)
    return {"rag": rag or "<<EMPTY>>"}

def router(state: AgentState):
    r = state.reasoning
    e = r.get("entities", {})

    if r["intent"] == "geospatial_request" and e.get("lon") and e.get("lat") and e.get("date"):
        return "call_mcp"
    return "no_mcp"

async def mcp_node(state: AgentState):
    r = state.reasoning["entities"]
    res = await call_mcp(r)
    return {"mcp": res}

def final_node(state: AgentState):
    user_msg = state.messages[-1]["content"]
    prompt = build_claude_prompt(
        user_msg=user_msg,
        reasoning=state.reasoning,
        rag=state.rag,
        mcp=state.mcp
    )
    answer = run_claude(prompt)
    return {"final_answer": answer}


###############################################
# BUILD GRAPH
###############################################

graph = StateGraph(AgentState)

graph.add_node("reasoning", reasoning_node)
graph.add_node("rag", rag_node)
graph.add_node("mcp", mcp_node)
graph.add_node("final", final_node)
graph.add_node("noop", lambda s: {})

graph.set_entry_point("reasoning")

graph.add_edge("reasoning", "rag")

graph.add_conditional_edges(
    "rag",
    router,
    {
        "call_mcp": "mcp",
        "no_mcp": "final",
    }
)

graph.add_edge("mcp", "final")
graph.add_edge("final", END)

workflow = graph.compile()


###############################################
# PUBLIC FUNCTION (FastAPI)
###############################################

async def run_agent(user_message: str, username: str):
    """
    Entry point used by FastAPI.
    """
    memory = get_memory(username)

    state = {
        "messages": [{"role": "user", "content": user_message}]
    }

    result = await workflow.ainvoke(state)

    # update memory
    memory.chat_memory.add_user_message(user_message)
    memory.chat_memory.add_ai_message(result["final_answer"])

    return {
        "answer": result["final_answer"],
        "reasoning": result["reasoning"],
        "rag_context": result["rag"],
        "mcp_result": result["mcp"],
    }
