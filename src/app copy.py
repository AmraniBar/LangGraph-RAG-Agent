import streamlit as st
import os
import time
from .main import setup_rag_system

st.set_page_config(
    page_title="Cybersecurity RAG Agent",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system once and cache it"""
    try:
        # Check if vector database exists
        import os

        persist_directory = "../data/chroma_db"
        if not os.path.exists(persist_directory):
            st.error(
                "Vector database not found! Please ensure the container was properly initialized."
            )
            return None

        return setup_rag_system()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None


def main():
    st.title("🔒 Cybersecurity RAG Agent")
    st.markdown("Ask questions about cybersecurity topics from NIST documents")

    # Configuration
    MAX_MESSAGES = 20  # Maximum number of conversation pairs to keep
    
    # Example Questions in main interface
    with st.expander("💡 Example Questions", expanded=False):
        example_questions = [
            "What are the six core functions of the NIST Cybersecurity Framework?",
            "How should organizations handle security incidents?",
            "What are the key security controls for access management?",
            "What is the incident response lifecycle?",
            "How to implement risk assessment procedures?",
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            col = cols[i % 2]
            with col:
                if st.button(f"📝 {question}", key=f"example_{i}", use_container_width=True):
                    st.session_state.pending_question = question
                    st.rerun()

    # Sidebar with information
    with st.sidebar:
        # Initialize sessions storage
        if "conversations" not in st.session_state:
            st.session_state.conversations = {}
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = "default"
            st.session_state.conversations["default"] = {"messages": [], "title": "New Conversation", "created": time.time()}
        
        # Session Management
        st.header("💬 Conversations")
        
        # New conversation button
        if st.button("➕ New Conversation", use_container_width=True):
            import uuid
            new_session_id = str(uuid.uuid4())[:8]
            st.session_state.conversations[new_session_id] = {
                "messages": [], 
                "title": "New Conversation", 
                "created": time.time()
            }
            st.session_state.current_session_id = new_session_id
            st.rerun()
        
        st.markdown("---")
        
        # Display conversation list
        for session_id, session_data in st.session_state.conversations.items():
            is_current = session_id == st.session_state.current_session_id
            
            # Create container for each conversation
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Conversation title/button
                    title = session_data.get("title", "New Conversation")
                    if len(session_data.get("messages", [])) > 0:
                        # Use first user message as title (truncated)
                        first_user_msg = next((msg["content"] for msg in session_data["messages"] if msg["role"] == "user"), title)
                        title = first_user_msg[:30] + "..." if len(first_user_msg) > 30 else first_user_msg
                    
                    button_type = "primary" if is_current else "secondary"
                    if st.button(f"{'🟢 ' if is_current else ''}🗨️ {title}", 
                               key=f"conv_{session_id}", 
                               type=button_type,
                               use_container_width=True):
                        st.session_state.current_session_id = session_id
                        st.rerun()
                
                with col2:
                    # Delete button
                    if len(st.session_state.conversations) > 1:  # Don't allow deleting the last conversation
                        if st.button("🗑️", key=f"del_{session_id}", help="Delete conversation"):
                            if session_id == st.session_state.current_session_id:
                                # Switch to another conversation before deleting
                                remaining_sessions = [sid for sid in st.session_state.conversations.keys() if sid != session_id]
                                st.session_state.current_session_id = remaining_sessions[0]
                            
                            del st.session_state.conversations[session_id]
                            st.rerun()
        
        st.markdown("---")
        st.header("About")
        st.markdown(
            """
        This application uses a Retrieval-Augmented Generation (RAG) system to answer 
        questions about cybersecurity based on NIST documents 
        
        **How to use:**
        1. Type your cybersecurity question in the chat input
        2. Press Enter or click "Send" 
        3. View the conversation history and responses with sources
        """
        )



    # Initialize the RAG system
    if "rag_app" not in st.session_state:
        with st.spinner("🔄 Loading RAG system..."):
            st.session_state.rag_app = initialize_rag_system()

    if st.session_state.rag_app is None:
        st.error(
            "❌ Failed to initialize the RAG system. Please check the configuration and try again."
        )
        return

    # Get current session messages
    current_session = st.session_state.conversations[st.session_state.current_session_id]
    messages = current_session["messages"]

    # Manage conversation length to prevent memory bloat
    if len(messages) > MAX_MESSAGES:
        # Keep only the most recent messages
        current_session["messages"] = messages[-MAX_MESSAGES:]
        messages = current_session["messages"]

    # Display chat messages from history
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "sources" in message:
                with st.expander(
                    f"📚 Sources ({len(message['sources'])} documents)", expanded=False
                ):
                    for i, source in enumerate(message["sources"], 1):
                        # Handle both old format (doc objects) and new format (dicts)
                        if isinstance(source, dict):
                            source_file = source.get("source", "Unknown source")
                            title = source.get("title", "Untitled")
                            page = source.get("page", "Unknown page")
                            content_preview = source.get("content", "")
                        else:
                            # Legacy format support
                            metadata = (
                                source.metadata if hasattr(source, "metadata") else {}
                            )
                            source_file = metadata.get("source", "Unknown source")
                            title = metadata.get("title", "Untitled")
                            page = metadata.get("page", "Unknown page")
                            content_preview = (
                                source.page_content[:200] + "..."
                                if len(source.page_content) > 200
                                else source.page_content
                            )

                        if source_file.startswith("documents/"):
                            source_file = source_file.replace("documents/", "")

                        if len(content_preview) > 200:
                            content_preview = content_preview[:200] + "..."

                        st.markdown(
                            f"""
                        **Source {i}:** {title}
                        
                        📁 **File:** {source_file} (Page {page})
                        
                        {content_preview}
                        """
                        )
                        if i < len(message["sources"]):
                            st.divider()

    # Chat input (always show)
    user_input = st.chat_input("Ask about cybersecurity...")

    # Check for pending question from example buttons or user input
    prompt = None
    if "pending_question" in st.session_state:
        prompt = st.session_state.pending_question
        del st.session_state.pending_question
    elif user_input:
        prompt = user_input

    if prompt:
        # Add user message with timestamp to current session
        current_session["messages"].append(
            {"role": "user", "content": prompt, "timestamp": time.time()}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Create placeholder for streaming response
            message_placeholder = st.empty()
            full_response = ""

            try:
                with st.spinner("Thinking..."):
                    # Invoke the RAG system
                    inputs = {"question": prompt}
                    result = st.session_state.rag_app.invoke(inputs)

                    # Process the answer
                    if "generation" in result and result["generation"]:
                        answer_content = str(result["generation"]).strip()

                        if (
                            answer_content.strip() == "question was not at all relevant"
                            or "I don't have the retrieved context" in answer_content
                        ):
                            answer = "⚠️ The question doesn't appear to be related to the cybersecurity documents in our knowledge base. Please ask questions about NIST cybersecurity frameworks, security controls, or incident handling."
                        elif (
                            not result.get("documents") or len(result["documents"]) == 0
                        ):
                            answer = "⚠️ No relevant documents were found for your question. Please try rephrasing or ask questions about NIST cybersecurity frameworks, security controls, or incident handling."
                        else:
                            answer = answer_content
                    else:
                        answer = "I couldn't generate a response. Please try rephrasing your question."

                # Simulate streaming by displaying response character by character
                for i in range(len(answer)):
                    full_response += answer[i]
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)  # Adjust speed of streaming

                # Final response without cursor
                message_placeholder.markdown(full_response)

                # Render sources if available
                sources = result.get("documents", [])
                if sources:
                    with st.expander(
                        f"📚 Sources ({len(sources)} documents)", expanded=False
                    ):
                        for i, doc in enumerate(sources, 1):
                            metadata = doc.metadata if hasattr(doc, "metadata") else {}
                            source_file = metadata.get("source", "Unknown source")
                            title = metadata.get("title", "Untitled")
                            page = metadata.get("page", "Unknown page")

                            if source_file.startswith("documents/"):
                                source_file = source_file.replace("documents/", "")

                            content_preview = (
                                doc.page_content[:200] + "..."
                                if len(doc.page_content) > 200
                                else doc.page_content
                            )

                            st.markdown(
                                f"""
                            **Source {i}:** {title}
                            
                            📁 **File:** {source_file} (Page {page})
                            
                            {content_preview}
                            """
                            )
                            if i < len(sources):
                                st.divider()

                # Add assistant message to history (optimized storage)
                assistant_message = {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": time.time(),
                }

                # Store sources more efficiently - only essential metadata
                if sources:
                    assistant_message["sources"] = [
                        {
                            "content": doc.page_content[
                                :500
                            ],  # Truncate content to save memory
                            "source": doc.metadata.get("source", "Unknown"),
                            "title": doc.metadata.get("title", "Untitled"),
                            "page": doc.metadata.get("page", "Unknown"),
                        }
                        for doc in sources
                    ]

                current_session["messages"].append(assistant_message)
                
                # Force refresh to update message counter
                st.rerun()

            except Exception as e:
                error_msg = "I encountered an error while processing your request."
                message_placeholder.markdown(error_msg)

                current_session["messages"].append(
                    {"role": "assistant", "content": error_msg, "timestamp": time.time()}
                )
                st.rerun()

    # # Footer
    # st.markdown("---")
    # st.markdown(
    #     "<p style='text-align: center; color: gray;'>Built with Streamlit • Powered by LangGraph RAG Agent</p>",
    #     unsafe_allow_html=True,
    # )


if __name__ == "__main__":
    main()
