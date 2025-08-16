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
    try:
        # Initialize embeddings
        embeddings = get_embeddings()

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

        if not existing_paths:
            return False

        # Load and process documents
        docs_list = load_documents(existing_paths)
        doc_splits = split_documents(docs_list)
        vectorstore = create_vectorstore(doc_splits, embeddings)

        # Verify the vector database
        try:
            collection_count = vectorstore._collection.count()
        except Exception as e:
            pass

        return True

    except Exception as e:
        return False


def main():
    """Main function to run vector database initialization"""
    # Check if vector database already exists and has data  
    from config import get_chroma_persist_directory
    persist_directory = get_chroma_persist_directory()
    if os.path.exists(persist_directory):
        try:
            from langchain_community.vectorstores import Chroma

            embeddings = get_embeddings()
            vectorstore = Chroma(
                collection_name="rag-chroma-optimized",
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )

            if vectorstore._collection.count() > 0:
                return True
        except Exception as e:
            pass

    # Initialize vector database
    success = initialize_vector_database()

    if success:
        return True
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
