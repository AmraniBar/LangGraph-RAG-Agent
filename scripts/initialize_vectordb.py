#!/usr/bin/env python3
"""
Standalone script to pre-initialize the vector database during container startup.
This ensures the vector DB is ready before the Streamlit app starts.
"""

import os
import sys
# Add both the project root and src directory to Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from config import get_embeddings
from data_preprocess.document_loader import (
    load_documents,
    split_documents,
    create_vectorstore,
)


def initialize_vector_database():
    """Initialize the vector database with all documents"""
    print("🔧 Starting vector database initialization...")

    try:
        # Initialize embeddings
        print("📊 Loading embeddings model...")
        embeddings = get_embeddings()
        print("✅ Embeddings model loaded successfully")

        # Document paths
        paths = [
            "data/documents/NIST.CSWP.29.pdf",
            "data/documents/NIST.SP.800-53r5.pdf",
            "data/documents/NIST.SP.800-61r3.pdf",
            "data/documents/nist.sp.800-61r2.pdf",
        ]

        # Check if documents exist
        existing_paths = []
        for path in paths:
            if os.path.exists(path):
                existing_paths.append(path)
                print(f"✅ Found document: {path}")
            else:
                print(f"⚠️  Document not found: {path}")

        if not existing_paths:
            print("❌ No documents found! Cannot initialize vector database.")
            return False

        # Load and process documents
        print("📖 Loading documents...")
        docs_list = load_documents(existing_paths)
        print(f"✅ Loaded {len(docs_list)} documents")

        print("✂️  Splitting documents into chunks...")
        doc_splits = split_documents(docs_list)
        print(f"✅ Created {len(doc_splits)} document chunks")

        print("🗄️  Creating vector database...")
        vectorstore = create_vectorstore(doc_splits, embeddings)
        print("✅ Vector database created successfully")

        # Verify the vector database
        try:
            collection_count = vectorstore._collection.count()
            print(f"✅ Vector database verified with {collection_count} documents")
        except Exception as e:
            print(f"⚠️  Could not verify collection count: {e}")

        print("🎉 Vector database initialization completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Vector database initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function to run vector database initialization"""
    print("=" * 60)
    print("🚀 RAG System Vector Database Initialization")
    print("=" * 60)

    # Check if vector database already exists and has data  
    from config import get_chroma_persist_directory
    persist_directory = get_chroma_persist_directory()
    if os.path.exists(persist_directory):
        print("🔍 Checking existing vector database...")
        try:
            from langchain_community.vectorstores import Chroma

            embeddings = get_embeddings()
            vectorstore = Chroma(
                collection_name="rag-chroma-optimized",
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )

            if vectorstore._collection.count() > 0:
                print(
                    f"✅ Vector database already exists with {vectorstore._collection.count()} documents"
                )
                print("🎉 Skipping initialization - vector database is ready!")
                return True
        except Exception as e:
            print(f"⚠️  Existing vector database check failed: {e}")
            print("🔧 Proceeding with fresh initialization...")

    # Initialize vector database
    success = initialize_vector_database()

    if success:
        print("\n" + "=" * 60)
        print("✅ VECTOR DATABASE READY FOR USE")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("❌ VECTOR DATABASE INITIALIZATION FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
