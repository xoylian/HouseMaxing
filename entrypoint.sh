#!/bin/bash
set -e

echo "🏠 HouseMaxing — Starting..."

# Run ingest if chroma_db doesn't exist yet
if [ ! -d "/app/chroma_db" ]; then
    echo "📚 Knowledge base not found. Running ingest..."
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "⚠️  OPENAI_API_KEY not set. Skipping ingest — enter key in the web UI."
    else
        # Try both lowercase and uppercase filename (case-sensitive Linux)
        INGEST_FILE=""
        if [ -f "/app/ingest.py" ]; then
            INGEST_FILE="/app/ingest.py"
        elif [ -f "/app/Ingest.py" ]; then
            INGEST_FILE="/app/Ingest.py"
        fi

        if [ -z "$INGEST_FILE" ]; then
            echo "⚠️  ingest.py not found in /app/"
        else
            python "$INGEST_FILE" && echo "✅ Ingest complete." || echo "⚠️  Ingest failed."
        fi
    fi
else
    echo "✅ Knowledge base already exists. Skipping ingest."
fi

echo "🚀 Starting Streamlit..."
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true