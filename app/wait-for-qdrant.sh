#!/bin/sh

QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"

echo "🔍 Waiting for Qdrant at $QDRANT_URL..."

# Try until Qdrant is ready
until curl -s -o /dev/null -w "%{http_code}" "$QDRANT_URL/readyz" | grep 200 > /dev/null; do
    echo "⏳ Qdrant not ready yet..."
    sleep 2
done

echo "✅ Qdrant is ready. Launching vectorstore builder..."
exec "$@"
