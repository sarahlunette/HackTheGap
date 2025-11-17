# TheLab - Crisis & Resilience Strategic Planner AI

A comprehensive RAG-powered crisis management and resilience planning system that combines Claude Sonnet 4.5, external MCP (Model Context Protocol) tools, and advanced geospatial analysis capabilities.

## 🏗️ Architecture

This system consists of two main components:
- **MCP Server**: External tool server providing OSM mapping and climate forecast capabilities
- **Main Application**: FastAPI-based RAG system with Claude integration and MCP client

## 📁 Repository Structure

```
TheLab_/
├── app/                    # Main application directory
│   ├── main_anthropic_mcp.py  # Primary application (RAG + Claude + MCP)
│   ├── main_anthropic.py      # Alternative version without MCP
│   ├── main.py                # Original version
│   ├── data/                   # Application data
│   ├── docs/                   # Documentation storage
│   ├── exports/                # Generated reports and exports
│   └── vectorstore/            # ChromaDB vector storage
├── mcp_server/             # MCP tools server
│   ├── main.py                # FastMCP server entry point
│   ├── tools/                 # Tool implementations
│   │   ├── osm_tool.py       # OpenStreetMap data tool
│   │   └── climate_tool.py   # Climate forecast tool
│   ├── requirements.txt      # MCP server dependencies
│   └── Dockerfile           # Container configuration
├── config/                 # Configuration files
├── data/                   # Shared data storage
├── docs/                   # Documentation
├── exports/               # Exported files
├── models/                # ML models
├── src/                   # Source code utilities
├── storage/               # File storage
├── vectorstore/          # Vector database storage
├── requirements.txt      # Main application dependencies
├── docker-compose.yml    # Container orchestration
└── .env                  # Environment variables
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Claude API Key (Anthropic)
- Required API keys for climate data (optional)

### Environment Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd TheLab_
```

2. **Set up environment variables**:
Create a `.env` file in the root directory:
```env
# Claude API
CLAUDE_API_KEY=your_anthropic_api_key_here
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Authentication
AUTH_MODE=basic
MVP_USER=admin
MVP_PASS=password

# MCP Server
MCP_WEBSOCKET_URL=ws://localhost:8100/ws
```

### 🛠️ Option 1: Docker Deployment (Recommended)

#### Launch MCP Server
```bash
# Build and start the MCP server
docker-compose up -d mcp

# Verify the server is running
curl http://localhost:8100/health
```

The MCP server will be available at `http://localhost:8100` with WebSocket endpoint at `ws://localhost:8100/ws`

#### Launch Main Application
```bash
# Install dependencies
pip install -r requirements.txt

# Navigate to app directory
cd app

# Start the FastAPI application
uvicorn main_anthropic_mcp:app --host 0.0.0.0 --port 8000 --reload
```

### 🖥️ Option 2: Local Development

#### Launch MCP Server Locally
```bash
# Navigate to MCP server directory
cd mcp_server

# Install dependencies
pip install -r requirements.txt

# Start the FastMCP server
python main.py
```

#### Launch Main Application
```bash
# Install main dependencies (from root directory)
pip install -r requirements.txt

# Navigate to app directory  
cd app

# Start the application
python -m uvicorn main_anthropic_mcp:app --host 0.0.0.0 --port 8000 --reload
```

## 📡 API Endpoints

### Main Application (Port 8000)

- **POST** `/chat` - Main chat interface with RAG and MCP integration
- **DELETE** `/chat/reset` - Reset conversation history
- **GET** `/plan?horizon=24` - Generate crisis response plan (24h or 72h)
- **POST** `/upload_doc` - Upload documents to knowledge base
- **GET** `/logs` - View system logs
- **GET** `/logs/export` - Export logs as CSV

### MCP Server (Port 8100)

- **WebSocket** `/ws` - MCP protocol endpoint
- **GET** `/health` - Health check
- Available tools:
  - `run_osm_data_tool` - OpenStreetMap data queries
  - `run_climate_forecast_tool` - Climate and weather forecasting

## 🔧 Usage Examples

### Basic Chat Request
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -u admin:password \
  -d '{"question": "What is the current infrastructure status in the coastal region?"}'
```

### Generate Crisis Plan
```bash
curl -X GET "http://localhost:8000/plan?horizon=24" \
  -u admin:password \
  --output crisis_plan_24h.pdf
```

### Upload Document
```bash
curl -X POST "http://localhost:8000/upload_doc" \
  -u admin:password \
  -F "file=@your_document.pdf"
```

## 🧠 Key Features

### 🤖 RESILIENCE-GPT AI Assistant
- **Crisis Response Planning**: Multi-phase disaster response (0-72h, weeks, months, years)
- **Geospatial Analysis**: GIS-informed infrastructure assessment
- **Multi-sector Coordination**: Energy, WASH, Health, Transport, Shelter, Communications
- **Long-context Processing**: Handles 50k-1M+ token contexts

### 📚 RAG (Retrieval-Augmented Generation)
- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: HuggingFace sentence-transformers
- **Document Processing**: PDF, DOCX, and text file support
- **Contextual Retrieval**: Similarity-based document search

### 🛠️ MCP Integration
- **External Tools**: Modular tool system via MCP protocol
- **Real-time Data**: Live OSM mapping and climate data
- **WebSocket Communication**: Efficient tool calling
- **Extensible Architecture**: Easy to add new tools

### 🗺️ Geospatial Capabilities
- **OpenStreetMap Integration**: Real-time infrastructure data
- **Climate Data Processing**: Weather and environmental analysis
- **Coordinate Systems**: Multi-projection support
- **Spatial Analysis**: Distance, routing, and accessibility calculations

## 📊 Output Formats

The system generates detailed crisis response plans with:

- **Executive Summaries** (600-1000 words)
- **Geospatial Segmentation** with risk zones
- **Priority Matrices** for resource allocation  
- **Sector-by-Sector Analysis** (Energy, WASH, Health, etc.)
- **Project Portfolios** with 15+ detailed projects
- **Financial Strategies** with cost breakdowns
- **Risk Registers** and mitigation plans
- **Multi-year Strategic Roadmaps**

## 🔒 Security & Authentication

- **Basic Authentication**: Username/password protection
- **Environment Variables**: Secure API key management
- **Request Logging**: Comprehensive audit trails
- **Input Validation**: Pydantic model validation

## 🧪 Development

### Adding New MCP Tools

1. Create tool in `mcp_server/tools/your_tool.py`:
```python
from fastmcp.tools import Tool

def your_new_tool(param1: str, param2: int) -> str:
    """Your tool description"""
    # Tool implementation
    return result

# Export as Tool
your_tool = Tool(your_new_tool)
```

2. Register in `mcp_server/main.py`:
```python
from tools.your_tool import your_new_tool

mcp = FastMCP(tools=[run_osm_data_tool, run_climate_forecast_tool, your_new_tool])
```

### Extending RAG Knowledge Base

- Add documents to `app/docs/` directory
- Use the `/upload_doc` endpoint
- Documents are automatically indexed and made searchable

## 📈 Performance & Scaling

- **Vector Database**: Persistent ChromaDB for fast similarity search
- **Async Processing**: FastAPI async endpoints
- **Memory Management**: Conversation buffer per user
- **Batch Processing**: Efficient document embedding
- **Caching**: Vector embeddings cached on disk

## 🐛 Troubleshooting

### Common Issues

1. **MCP Connection Failed**:
   - Verify MCP server is running on port 8100
   - Check WebSocket URL in environment variables
   - Ensure Docker containers are healthy

2. **Claude API Errors**:
   - Validate CLAUDE_API_KEY in .env file
   - Check API rate limits and usage
   - Verify model name (claude-3-5-sonnet-20241022)

3. **Vector Store Issues**:
   - Clear `vectorstore/chroma` directory to reset
   - Check write permissions in vectorstore directory
   - Verify HuggingFace model downloads

4. **Authentication Problems**:
   - Verify MVP_USER and MVP_PASS in .env
   - Use correct basic auth headers
   - Check AUTH_MODE setting

### Logs and Monitoring

- Application logs via `/logs` endpoint
- Docker container logs: `docker-compose logs mcp`
- FastAPI debug mode with `--reload` flag

## 📝 License

This project is part of TheLab - Crisis & Resilience AI research initiative.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

## 📞 Support

For issues and questions:
- Check troubleshooting section
- Review application logs
- Create GitHub issue with detailed description
