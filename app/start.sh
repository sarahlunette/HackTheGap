#!/bin/bash
set -e

# -----------------------------
#  Configuration
# -----------------------------
QDRANT_URL="$qdrant_url"        # e.g., https://YOUR-CLUSTER.qdrant.io
QDRANT_API_KEY="$qdrant_api_key"
COLLECTION="island_docs"

echo "⏳ Waiting for Qdrant at: $QDRANT_URL"

# -----------------------------
#  Wait for Qdrant to be ready
# -----------------------------
MAX_RETRIES=120
COUNTER=0

while true; do
    RESPONSE=$(curl -s -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/collections" || true)

    # Check if response is valid JSON
    if echo "$RESPONSE" | jq empty >/dev/null 2>&1; then
        echo "🟢 Qdrant collections endpoint is ready."
        break
    fi

    COUNTER=$((COUNTER+1))
    if [[ $COUNTER -gt $MAX_RETRIES ]]; then
        echo "❌ Qdrant did not become ready after $MAX_RETRIES seconds."
        exit 1
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

    curl -X PUT "$QDRANT_URL/collections/$COLLECTION" \
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
#  Run vectorstore builder
# -----------------------------
echo "🚀 Running vectorstore initialization…"

if python build_vectorstore.py; then
    echo "🟢 Vectorstore ready."
else
    echo "⚠ Vectorstore build failed — continuing anyway."
fi

# -----------------------------
#  Start the API
# -----------------------------
echo "🚀 Starting FastAPI server…"

exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --reload
