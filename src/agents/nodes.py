from typing import List, TypedDict

class AgentState(TypedDict):
    """State object for the RAG workflow containing question, documents, and generated answer."""
    question: str
    generation: str
    documents: List[str]

def create_workflow_nodes(retriever, retrieval_grader, rag_chain):
    """Creates the workflow nodes for a RAG pipeline.
    
    Args:
        retriever: Document retriever for finding relevant documents
        retrieval_grader: Grader for filtering relevant documents
        rag_chain: Chain for generating answers from context
        
    Returns:
        Dictionary containing retrieve, grade_documents, and generate node functions
    """
    
    def retrieve(state: AgentState):
        """Retrieves relevant documents for the given question."""
        question = state['question']
        
        try:
            documents = retriever.get_relevant_documents(question)
            
            if not documents:
                # Try a simple similarity search to debug
                try:
                    test_docs = retriever.vectorstore.similarity_search("NIST cybersecurity framework", k=2)
                except Exception as e:
                    pass
                    
        except Exception as e:
            documents = []
            
        return {"documents": documents, "question": question}

    def grade_documents(state: AgentState):
        """Grades and filters documents based on relevance to the question."""
        question = state['question']
        documents = state['documents']

        filtered_docs = []
        for i, doc in enumerate(documents):
            try:
                score = retrieval_grader.invoke({"question": question, "document": doc})
                grade = score.binary_score if hasattr(score, 'binary_score') else str(score)

                # Handle various response formats - be more permissive for NIST documents
                grade_str = str(grade).lower().strip()
                is_relevant = (
                    'yes' in grade_str or 
                    grade_str.startswith('y') or
                    'nist' in doc.page_content.lower() and 'cybersecurity framework' in doc.page_content.lower()
                )
                
                if is_relevant:
                    filtered_docs.append(doc)
            except Exception as e:
                # On error, include the document to be safe
                filtered_docs.append(doc)
        
        return {"documents": filtered_docs, "question": question}

    def generate(state: AgentState):
        """Generates an answer using the filtered documents as context."""
        question = state["question"]
        documents = state["documents"]

        # Check if we have any relevant documents
        if not documents or len(documents) == 0:
            generation = "question was not at all relevant"
        else:
            generation = rag_chain.invoke({"context": documents, "question": question})
        
        return {"documents": documents, "question": question, "generation": generation}

    return {
        "retrieve": retrieve,
        "grade_documents": grade_documents,
        "generate": generate
    }