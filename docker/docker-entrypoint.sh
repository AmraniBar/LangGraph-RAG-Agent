#!/bin/bash

# Docker entrypoint script for RAG Agent

set -e

echo "🐳 Starting RAG Agent Docker Container..."

# Start Ollama service in background
echo "🚀 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
while ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; do
    sleep 2
    echo "   Still waiting for Ollama..."
done
echo "✅ Ollama is ready!"

# Check if llama3.2 model exists, if not pull it
echo "🔍 Checking for llama3.2 model..."
if ! ollama list | grep -q "llama3.2"; then
    echo "📥 Pulling llama3.2 model (this may take several minutes)..."
    ollama pull llama3.2
    echo "✅ llama3.2 model ready!"
else
    echo "✅ llama3.2 model already available!"
fi

# Create ChromaDB directory if it doesn't exist
mkdir -p data/chroma_db
echo "📁 Current directory: $(pwd)"

# Add src to Python path so relative imports work
export PYTHONPATH="/app:/app/src:$PYTHONPATH"

# Pre-initialize vector database (skip if script doesn't exist)
if [ -f "scripts/initialize_vectordb.py" ]; then
    echo "🗄️  Initializing vector database..."
    python scripts/initialize_vectordb.py
    if [ $? -ne 0 ]; then
        echo "⚠️  Vector database initialization failed! Continuing anyway..."
    fi
else
    echo "⚠️  Vector database initialization script not found. Skipping..."
fi

# Set environment variables for Streamlit
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "🌐 Starting Streamlit application..."
echo "   Access the application at: http://localhost:8501"
echo "📁 Current directory: $(pwd)"
# Start Streamlit from project root
echo "📁 Current directory: $(pwd)"
echo "📂 Checking for src/app.py:"
if [ -f "src/app.py" ]; then
    echo "✅ Found src/app.py"
else
    echo "❌ src/app.py not found!"
fi
echo "🔧 Testing Ollama connection..."
curl -f http://localhost:11434/api/tags || echo "⚠️ Ollama connection test failed"
echo "🚀 Starting Streamlit..."
exec streamlit run src/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false