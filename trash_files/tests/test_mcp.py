import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from mcp_use import MCPAgent, MCPClient

CONFIG = {
    "mcpServers": {
        "public-demo-fastapi": {
            "url": "http://localhost:9001/mcp"
        }
    }
}

app = FastAPI(title="Claude Agent Orchestrator")

class AgentRequest(BaseModel):
    prompt: str

@app.post("/agent")
async def run_agent(req: AgentRequest):
    # Async context for each request (stateless HTTP API) 
    async def agent_call():
        client = MCPClient.from_dict(CONFIG)
        llm = ChatAnthropic(model="claude-sonnet-4.5", api_key=os.getenv("CLAUDE_API_KEY"))
        agent = MCPAgent(llm=llm, client=client, max_steps=20)
        result = await agent.run(req.prompt)
        await client.close_all_sessions()
        return result

    output = await agent_call()
    return {"result": output}

# To launch: uvicorn agent_api:app --port 11400 --reload
