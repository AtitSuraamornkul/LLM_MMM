import streamlit as st

def initialize_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        "chat_history": [],
        "display_history": [],
        "welcome_shown" : False,
        "total_tokens_used": 0,
        "request_count": 0,
        "uploaded_context": "",
        "welcome_shown": False,
        "show_modal": False,
        "html_content": "",
        "max_chat_history": 2,  # Limit chat history
        "max_context_tokens": 5000,  # Limit context size
        "retriever": None  # For RAG retriever caching
    }
    # In utils/cleanup.py, add this to initialize_session_state()
    if 'complexity_level' not in st.session_state:
        st.session_state.complexity_level = 2  # Default to 2
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def cleanup_chat_history():
    """Keep only the last few messages based on session state setting"""
    MAX_MESSAGES = st.session_state.max_chat_history
    
    if len(st.session_state.chat_history) > MAX_MESSAGES:
        old_count = len(st.session_state.chat_history)
        st.session_state.chat_history = []
        st.info(f"🧹 Cleaned up chat history")


def clear_on_complexity_change():
    """Clear chat history when complexity level changes"""
    if hasattr(st.session_state, 'previous_complexity_level'):
        if st.session_state.complexity_level != st.session_state.previous_complexity_level:
            old_count = len(st.session_state.chat_history)
            st.session_state.chat_history = []
            
            level_names = {1: "Basic", 2: "Medium", 3: "Advanced"}
            old_level = level_names.get(st.session_state.previous_complexity_level, "Unknown")
            new_level = level_names.get(st.session_state.complexity_level, "Unknown")
            
            if old_count > 0:  # Only show message if there was history to clear
                st.success(f"Switched to {new_level} style - Chat cleared for fresh start!")
    
    st.session_state.previous_complexity_level = st.session_state.complexity_level