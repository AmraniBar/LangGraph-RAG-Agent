#!/usr/bin/env python3
"""
Debug script to test retrieval system independently
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import setup_rag_system

def test_retrieval():
    print("=== TESTING RETRIEVAL SYSTEM ===")
    
    try:
        print("1. Setting up RAG system...")
        app = setup_rag_system()
        print("✓ RAG system initialized")
        
        print("\n2. Testing question: 'What are the five core functions of the NIST Cybersecurity Framework?'")
        question = "What are the five core functions of the NIST Cybersecurity Framework?"
        
        inputs = {"question": question}
        result = app.invoke(inputs)
        
        print(f"\n3. Result keys: {list(result.keys())}")
        
        if "documents" in result:
            docs = result["documents"]
            print(f"\n4. Retrieved {len(docs)} documents:")
            for i, doc in enumerate(docs):
                print(f"\n--- DOCUMENT {i+1} ---")
                print(f"Content: {doc.page_content[:300]}...")
                if hasattr(doc, 'metadata'):
                    print(f"Metadata: {doc.metadata}")
        
        if "generation" in result:
            print(f"\n5. Generation result:")
            print(f"Type: {type(result['generation'])}")
            print(f"Content: {result['generation']}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval()