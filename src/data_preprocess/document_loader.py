from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from typing import List
import os
import time
import pickle
from data_preprocess.header_footer_cleaner import clean_chunked_documents


def load_documents(paths: List[str]):
    """Load documents with caching"""
    cache_file = "data/cache/cached_documents.pkl"
    
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # Check if cached version exists and is newer than source files
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        source_times = [
            os.path.getmtime(path) for path in paths if os.path.exists(path)
        ]

        if source_times and max(source_times) < cache_time:
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                pass

    start_time = time.time()
    docs_list = []

    for path in paths:
        if os.path.exists(path):
            docs = PyPDFLoader(path).load()
            docs_list.extend(docs)
        else:
            pass

    # Cache the loaded documents
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(docs_list, f)
    except Exception as e:
        pass

    end_time = time.time()
    return docs_list


def split_documents_optimized(docs_list, clean_headers_footers=True):
    """Optimized document splitting with smaller chunks for better retrieval"""
    cache_file = "data/cache/cached_document_splits.pkl"
    
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # Check if cached splits exist
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            pass

    start_time = time.time()

    text_splitter = SentenceTransformersTokenTextSplitter(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=400,  # Smaller chunks for better precision
        chunk_overlap=25,  # Reduced overlap for faster processing
    )

    doc_splits = text_splitter.split_documents(docs_list)

    # Clean headers and footers if requested
    if clean_headers_footers:
        doc_splits = clean_chunked_documents(doc_splits)

    # Cache the splits
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(doc_splits, f)
    except Exception as e:
        pass

    end_time = time.time()
    return doc_splits


def create_vectorstore_persistent(doc_splits, embeddings):
    """Create persistent vector store to avoid reprocessing"""
    # Use centralized path configuration to prevent multiple folders
    from config import get_chroma_persist_directory
    persist_directory = get_chroma_persist_directory()
    collection_name = "rag-chroma-optimized"

    start_time = time.time()

    # Try to load existing vectorstore
    if os.path.exists(persist_directory):
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )
            # Check if it has documents
            if vectorstore._collection.count() > 0:
                return vectorstore
        except Exception as e:
            pass

    # Validate that we have documents to process
    if not doc_splits:
        raise ValueError("Cannot create vector store: No documents provided for embedding")
    
    # Filter out any empty documents
    valid_docs = [doc for doc in doc_splits if doc.page_content.strip()]
    
    if not valid_docs:
        raise ValueError("Cannot create vector store: All documents are empty after filtering")
    
    vectorstore = Chroma.from_documents(
        documents=valid_docs,
        collection_name=collection_name,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    end_time = time.time()
    return vectorstore


def setup_optimized_retriever_tool(vectorstore):
    """Setup retriever with optimized parameters"""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},  # Retrieve fewer documents for faster processing
    )

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_cybersecurity_documents",
        "Search and return information about cybersecurity content from documents.",
    )
    return retriever, retriever_tool


# Legacy function names for backward compatibility
def split_documents(docs_list, clean_headers_footers=True):
    """Backward compatibility wrapper"""
    return split_documents_optimized(docs_list, clean_headers_footers)


def create_vectorstore(doc_splits, embeddings):
    """Backward compatibility wrapper"""
    return create_vectorstore_persistent(doc_splits, embeddings)


def setup_retriever_tool(vectorstore):
    """Backward compatibility wrapper"""
    return setup_optimized_retriever_tool(vectorstore)
