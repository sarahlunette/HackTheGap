#!/bin/bash
set -e

QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
COLLECTION="island_docs"

echo "⏳ Waiting for Qdrant at: $QDRANT_URL"

# Maximum wait time: 120 seconds
MAX_RETRIES=120
COUNTER=0

while true; do
    # Check endpoint
    if curl -s "$QDRANT_URL/collections" >/dev/null 2>&1; then
        echo "✅ Qdrant API is responding."

        # Now ensure Qdrant is REALLY ready (not loading shards)
        STATUS=$(curl -s "$QDRANT_URL/collections" | grep -o "\"status\":" || true)
        if [[ ! -z "$STATUS" ]]; then
            echo "🟢 Qdrant collections endpoint operational."
            break
        fi
    fi

    COUNTER=$((COUNTER+1))
    if [[ $COUNTER -gt $MAX_RETRIES ]]; then
        echo "❌ Qdrant did not become ready after $MAX_RETRIES seconds."
        exit 1
    fi

    echo "   …waiting ($COUNTER/$MAX_RETRIES)"
    sleep 1
done


# -------------------------------------------------------------
#  Ensure target collection exists (safe to run multiple times)
# -------------------------------------------------------------
echo "📁 Checking if collection '$COLLECTION' exists…"

EXISTS=$(curl -s "$QDRANT_URL/collections/$COLLECTION/exists" | jq -r '.result.exists')

if [[ "$EXISTS" == "true" ]]; then
    echo "✔ Collection already exists."
else
    echo "⚠ Collection missing — creating '$COLLECTION'…"

    # IMPORTANT: embedding size must match your model
    curl -X PUT "$QDRANT_URL/collections/$COLLECTION" \
      -H "Content-Type: application/json" \
      --data '{
        "vectors": {
          "size": 384,
          "distance": "Cosine"
        }
      }'

    echo "✔ Collection created."
fi


# -------------------------------------------------------------
#  Run vectorstore builder
# -------------------------------------------------------------
echo "🚀 Running vectorstore initialization…"

if python build_vectorstore.py; then
    echo "🟢 Vectorstore ready."
else
    echo "⚠ Vectorstore build failed — continuing anyway."
fi


# -------------------------------------------------------------
#  Start the API
# -------------------------------------------------------------
echo "🚀 Starting FastAPI server…"

exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
