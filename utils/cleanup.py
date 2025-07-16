import streamlit as st

def initialize_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        "chat_history": [],
        "total_tokens_used": 0,
        "request_count": 0,
        "uploaded_context": "",
        "welcome_shown": False,
        "show_modal": False,
        "html_content": "",
        "max_chat_history": 50,  # Limit chat history
        "max_context_tokens": 8000,  # Limit context size
        "retriever": None  # For RAG retriever caching
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def cleanup_chat_history():
    """Auto-cleanup old chat messages"""
    if len(st.session_state.chat_history) > st.session_state.max_chat_history:
        # Keep only recent messages (last 50)
        old_count = len(st.session_state.chat_history)
        st.session_state.chat_history = st.session_state.chat_history[-st.session_state.max_chat_history:]
        st.info(f"🧹 Cleaned up chat history: {old_count} → {len(st.session_state.chat_history)} messages")


