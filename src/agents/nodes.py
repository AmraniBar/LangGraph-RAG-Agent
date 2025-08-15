from typing import List, TypedDict

class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def create_workflow_nodes(retriever, retrieval_grader, rag_chain):
    
    def retrieve(state: AgentState):
        print("----RETRIEVE----", flush=True)
        question = state['question']
        print(f"Question: '{question}'", flush=True)
        
        try:
            documents = retriever.get_relevant_documents(question)
            print(f"Retrieved {len(documents)} documents", flush=True)
            
            if documents:
                for i, doc in enumerate(documents):
                    print(f"----DOCUMENT {i+1}----", flush=True)
                    print(f"Content preview: {doc.page_content[:500]}...", flush=True)
                    if hasattr(doc, 'metadata'):
                        print(f"Metadata: {doc.metadata}", flush=True)
                    print("----END DOCUMENT----", flush=True)
            else:
                print("----NO DOCUMENTS RETRIEVED - CHECKING VECTOR STORE----", flush=True)
                # Try a simple similarity search to debug
                try:
                    test_docs = retriever.vectorstore.similarity_search("NIST cybersecurity framework", k=2)
                    print(f"Test search for 'NIST cybersecurity framework' returned {len(test_docs)} docs", flush=True)
                    if test_docs:
                        print(f"Test doc preview: {test_docs[0].page_content[:200]}...", flush=True)
                except Exception as e:
                    print(f"Test search failed: {e}", flush=True)
                    
        except Exception as e:
            print(f"----RETRIEVAL ERROR: {e}----", flush=True)
            documents = []
            
        return {"documents": documents, "question": question}

    def grade_documents(state: AgentState):
        print("----CHECK DOCUMENTS RELEVANCE TO THE QUESTION----", flush=True)
        question = state['question']
        documents = state['documents']
        print(f"Grading {len(documents)} documents", flush=True)

        filtered_docs = []
        for i, doc in enumerate(documents):
            print(f"----GRADING DOCUMENT {i+1}----", flush=True)
            print(f"Document preview: {doc.page_content[:200]}...", flush=True)
            
            try:
                score = retrieval_grader.invoke({"question": question, "document": doc})
                grade = score.binary_score if hasattr(score, 'binary_score') else str(score)
                print(f"Grader returned: '{grade}'", flush=True)

                # Handle various response formats - be more permissive for NIST documents
                grade_str = str(grade).lower().strip()
                is_relevant = (
                    'yes' in grade_str or 
                    grade_str.startswith('y') or
                    'nist' in doc.page_content.lower() and 'cybersecurity framework' in doc.page_content.lower()
                )
                
                if is_relevant:
                    print(f"----GRADE: DOCUMENT {i+1} RELEVANT----", flush=True)
                    filtered_docs.append(doc)
                else:
                    print(f"----GRADE: DOCUMENT {i+1} NOT RELEVANT----", flush=True)
            except Exception as e:
                print(f"----GRADING ERROR for document {i+1}: {e}----", flush=True)
                # On error, include the document to be safe
                filtered_docs.append(doc)
        
        print(f"Filtered documents: {len(filtered_docs)} out of {len(documents)}", flush=True)
        return {"documents": filtered_docs, "question": question}

    def generate(state: AgentState):
        print("----GENERATE----", flush=True)
        question = state["question"]
        documents = state["documents"]
        print(f"Generating answer for question: '{question}'", flush=True)
        print(f"Using {len(documents)} documents for context", flush=True)

        # Check if we have any relevant documents
        if not documents or len(documents) == 0:
            generation = "question was not at all relevant"
            print("No documents - returning 'not relevant'", flush=True)
        else:
            print("Invoking RAG chain with documents...", flush=True)
            # Show context being sent to LLM
            context_preview = documents[0].page_content[:200] if documents else "No context"
            print(f"Context preview: {context_preview}...", flush=True)
            
            generation = rag_chain.invoke({"context": documents, "question": question})
            print(f"RAG chain returned: {type(generation)}", flush=True)
            print(f"Generation content: '{generation}'", flush=True)
            print(f"Generation length: {len(generation) if generation else 0}", flush=True)
        
        print("----GENERATE COMPLETE----", flush=True)
        return {"documents": documents, "question": question, "generation": generation}

    return {
        "retrieve": retrieve,
        "grade_documents": grade_documents,
        "generate": generate
    }