import os
import logging
import json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from collections import defaultdict

# LangChain/Claude agent support
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import Tool
from langchain.agents import initialize_agent, AgentType # v0.1+ only
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

# MCP Agent
from mcp_use import MCPAgent, MCPClient

# Your tool imports
from tools.earth_engine_tool import fetch_earth_engine_data
from tools.climate_tool import fetch_climate_data

# Other libraries as needed (RAG, Qdrant, etc.)
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

app = FastAPI(title="LangChain Claude + MCPAgent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- AUTH ----
AUTH_MODE = os.getenv("AUTH_MODE", "basic")
MVP_USER = os.getenv("MVP_USER", "admin")
MVP_PASS = os.getenv("MVP_PASS", "password")
security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != MVP_USER or credentials.password != MVP_PASS:
        raise HTTPException(401, "Unauthorized")
    return credentials.username

# ---- RAG setup ----
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "island_docs")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant_client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(
    client=qdrant_client, collection_name=COLLECTION_NAME,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, storage_context=storage_context, embed_model=embed_model,
)
query_engine = index.as_retriever(similarity_top_k=3)
USER_MEMORIES = defaultdict(
    lambda: ConversationBufferMemory(
        return_messages=True,
        chat_memory=ChatMessageHistory()
    )
)

def query_knowledge_base(question: str) -> str:
    try:
        nodes = query_engine.retrieve(question)
    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        return ""
    return "\n".join(n.text for n in nodes)

class ChatRequest(BaseModel):
    question: str

# ---- MCPAgent LangChain Tool Wrapper ----
CONFIG = {
    "mcpServers": {
        "public-demo-fastapi": {"url": "http://localhost:9001/mcp"}
    }
}
llm = ChatAnthropic(model="claude-3-sonnet-20240229", api_key=os.getenv("CLAUDE_API_KEY"))
client = MCPClient.from_dict(CONFIG)
agent_instance = MCPAgent(llm=llm, client=client, max_steps=20) # Unused below, for direct calls only

def mcp_agent_tool_func(prompt: str) -> str:
    """LangChain Tool function: calls MCPAgent on prompt."""
    # The agent might be async, so wrap in event loop if needed
    import asyncio
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(agent_instance.run(prompt))
    return json.dumps(result)

mcp_agent_tool = Tool(
    name="mcp_agent",
    func=mcp_agent_tool_func,
    description="Runs the MCP agent workflow with the provided prompt, accessing Earth Engine, climate, and geospatial tools."
)

# ---- Other tool wrappers if you want ----
def earth_engine_tool_func(lon: float, lat: float, date: str, radius: int):
    return fetch_earth_engine_data(lon=lon, lat=lat, date=date, radius=radius)
def climate_tool_func(location: str, date: str):
    return fetch_climate_data(location=location, date=date)

# ---- LangChain AGENT ----
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [mcp_agent_tool]
langchain_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS, # Choose the Claude/Function-calling compatible agent here, e.g. OPENAI_FUNCTIONS or ANTHROPIC_FUNCTIONS
    memory=memory,
    verbose=True,
)

@app.post("/chat")
async def chat(req: ChatRequest, username: str = Depends(verify_credentials)):
    user_msg = req.question.strip()
    memory = USER_MEMORIES[username]

    # Optionally include RAG context retrieval
    rag_context = query_knowledge_base(user_msg)
    rag_block = rag_context if rag_context.strip() else "<<EMPTY>>"

    # Compose system message or context (Claude history, persona, RAG, etc.)
    system_prefix = f"""
You are Crisis Resilience-GPT, an agent for post-disaster resilience planning.
RAG CONTEXT:
{rag_block}
User message:
"""
    lc_input = system_prefix + user_msg

    # Run LangChain Agent, which calls MCPAgent tool directly if needed
    response = langchain_agent.run(lc_input)

    # Optionally add output to your own conversation memory
    memory.chat_memory.add_user_message(user_msg)
    memory.chat_memory.add_ai_message(response)

    return {
        "answer": response,
        "context_used": rag_block,
        "conversation_turns": len(memory.chat_memory.messages) // 2,
    }

@app.delete("/chat/reset")
def reset_history(username: str = Depends(verify_credentials)):
    USER_MEMORIES[username] = ConversationBufferMemory(
        return_messages=True,
        chat_memory=ChatMessageHistory()
    )
    return {"message": "Memory cleared."}

# Optionally: keep your /agent endpoint for direct MCPAgent calls
class AgentRequest(BaseModel):
    prompt: str

@app.post("/agent")
async def run_agent(req: AgentRequest, username: str = Depends(verify_credentials)):
    result = await agent_instance.run(req.prompt)
    await client.close_all_sessions()
    return {"result": result}

# Optionally, add MCP API mounting if you want tool exposure via /mcp
from fastapi_mcp import FastApiMCP
mcp = FastApiMCP(app)
mcp.mount()
