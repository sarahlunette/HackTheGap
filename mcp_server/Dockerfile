# ============================
#  MCP SERVER
# ============================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# global requirements
COPY  requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Copy all MCP server code
COPY . /app

EXPOSE 9001

# Run the MCP server
CMD ["python", "main.py"]
