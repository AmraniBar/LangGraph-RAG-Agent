import os
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Centralized path configuration to prevent multiple chroma_db folders
def get_chroma_persist_directory():
    """Get the absolute path for ChromaDB persistence directory"""
    # Always resolve relative to /app in Docker container
    if os.path.exists('/app'):
        # Running in Docker container
        persist_dir = '/app/data/chroma_db'
    else:
        # Running locally - use current working directory
        persist_dir = os.path.abspath('data/chroma_db')
    
    # Ensure the directory exists
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm():
    return ChatOllama(model="llama3.2")