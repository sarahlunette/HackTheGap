"""
FASTAPI frontend for the LangGraph Agent (agent.py)
---------------------------------------------------

This file ONLY:
- exposes /chat via HTTP
- authenticates user
- forwards the request to run_agent() from agent.py
- returns the structured response from the agent

All reasoning, MCP tool-calls, RAG, memory, and Claude synthesis
are handled inside agent.py — the brain of the system.
"""

import os
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from dotenv import load_dotenv

# Import your unified LangGraph agent
from agent import run_agent

# Load environment
load_dotenv()

AUTH_MODE = os.getenv("AUTH_MODE", "basic")
MVP_USER = os.getenv("MVP_USER", "admin")
MVP_PASS = os.getenv("MVP_PASS", "password")

app = FastAPI(title="ResilienceGPT — Unified LangGraph Agent API")
security = HTTPBasic()


# ------------------------------------------------------
# Authentication logic
# ------------------------------------------------------
def verify_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
    authorization: str = Header(None),
):

    if AUTH_MODE == "google":
        # Expect a Google Bearer token (Cloud Run OIDC)
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing Bearer token")
        return authorization.split(" ")[1]

    # Basic auth (default)
    if credentials.username != MVP_USER or credentials.password != MVP_PASS:
        raise HTTPException(401, "Unauthorized")

    return credentials.username


# ------------------------------------------------------
# Request schema
# ------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


# ------------------------------------------------------
# Main Chat Endpoint (Forward to LangGraph)
# ------------------------------------------------------
@app.post("/chat")
async def chat(req: ChatRequest, username: str = Depends(verify_credentials)):
    """
    Forwards the user's question to the LangGraph agent.
    The agent handles:
      - reasoning via Mistral
      - RAG query
      - MCP call (if needed)
      - final synthesis via Claude
      - memory per user
    """

    result = await run_agent(
        user_message=req.question,
        username=username
    )

    return {
        "answer": result["answer"],
        "reasoning": result["reasoning"],
        "rag_context": result["rag_context"],
        "mcp_result": result["mcp_result"],
        "user": username,
    }


# ------------------------------------------------------
# Reset user memory
# ------------------------------------------------------
@app.delete("/chat/reset")
async def reset_memory(username: str = Depends(verify_credentials)):
    """
    Optional: Clears memory for a user inside agent.py
    """
    from agent import USER_MEMORIES
    USER_MEMORIES[username] = None
    return {"message": f"Memory reset for user {username}"}


# ------------------------------------------------------
# Health check (Cloud Run)
# ------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
