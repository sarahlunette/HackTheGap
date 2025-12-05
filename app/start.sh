#!/bin/bash
set -e

# -----------------------------
#  Configuration from env vars
# -----------------------------
QDRANT_URL="${QDRANT_URL:?QDRANT_URL is not set}"
QDRANT_API_KEY="${QDRANT_API_KEY:?QDRANT_API_KEY is not set}"
COLLECTION="island_docs"

PORT="${PORT:-8080}"

echo "💡 Cloud Run PORT is: $PORT"
echo "💡 Qdrant URL: $QDRANT_URL"

# -----------------------------
#  Start FastAPI immediately
# -----------------------------
echo "🚀 Starting FastAPI server in background..."
uvicorn main:app --host 0.0.0.0 --port $PORT &

FASTAPI_PID=$!

# -----------------------------
#  Wait for Qdrant to be ready
# -----------------------------
MAX_RETRIES=120
COUNTER=0

echo "⏳ Waiting for Qdrant collections endpoint..."
while true; do
    RESPONSE=$(curl -s -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/collections" || true)
    if echo "$RESPONSE" | jq empty >/dev/null 2>&1; then
        echo "🟢 Qdrant collections endpoint is ready."
        break
    fi
    COUNTER=$((COUNTER+1))
    if [[ $COUNTER -gt $MAX_RETRIES ]]; then
        echo "❌ Qdrant did not become ready after $MAX_RETRIES seconds."
        break
    fi
    echo "   …waiting ($COUNTER/$MAX_RETRIES)"
    sleep 1
done

# -----------------------------
#  Ensure target collection exists
# -----------------------------
echo "📁 Checking if collection '$COLLECTION' exists…"
EXISTS=$(curl -s -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/collections/$COLLECTION/exists" | jq -r '.result.exists // false')

if [[ "$EXISTS" == "true" ]]; then
    echo "✔ Collection already exists."
else
    echo "⚠ Collection missing — creating '$COLLECTION'…"
    curl -s -X PUT "$QDRANT_URL/collections/$COLLECTION" \
      -H "Content-Type: application/json" \
      -H "api-key: $QDRANT_API_KEY" \
      --data '{
        "vectors": {
          "size": 384,
          "distance": "Cosine"
        }
      }'
    echo "✔ Collection created."
fi

# -----------------------------
#  Build vector store in background
# -----------------------------
echo "🚀 Running vectorstore initialization in background..."
(
    if python build_vectorstore.py; then
        echo "🟢 Vectorstore ready."
    else
        echo "⚠ Vectorstore build failed — continuing anyway."
    fi
) &

# -----------------------------
#  Wait for FastAPI process to keep container alive
# -----------------------------
wait $FASTAPI_PID
