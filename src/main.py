from config import get_embeddings, get_llm
from data_preprocess.document_loader import (
    load_documents,
    split_documents,
    create_vectorstore,
    setup_retriever_tool,
)
from agents.graders import create_document_grader
from agents.chains import create_rag_chain
from agents.nodes import create_workflow_nodes
from agents.graph import create_workflow


def setup_rag_system():
    # Initialize models
    embeddings = get_embeddings()
    llm = get_llm()

    # Load and process documents
    paths = [
        "data/documents/NIST.CSWP.29.pdf",
        "data/documents/NIST.SP.800-53r5.pdf",
        "data/documents/NIST.SP.800-61r3.pdf",
        "data/documents/nist.sp.800-61r2.pdf",
    ]

    docs_list = load_documents(paths)
    doc_splits = split_documents(docs_list)
    vectorstore = create_vectorstore(doc_splits, embeddings)
    retriever, retriever_tool = setup_retriever_tool(vectorstore)

    # Create graders and chains
    retrieval_grader = create_document_grader(llm)
    rag_chain = create_rag_chain(llm)

    # Create workflow nodes
    nodes = create_workflow_nodes(retriever, retrieval_grader, rag_chain)

    # Create and compile workflow
    app = create_workflow(nodes)

    return app


def main():
    app = setup_rag_system()


if __name__ == "__main__":
    main()
