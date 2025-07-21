import streamlit as st
import utils.check_token as check_token
from utils.file_processing import analyze_csv_files_dynamically

MAX_TOKENS = 10000
RESERVED_FOR_ANSWER = 2000 

def get_rag_context(query, retriever):
    """Retrieve relevant context using RAG, trimmed to fit token limit."""
    try:
        results = retriever.invoke(query)
        if not results:
            return "No relevant context found in the knowledge base."
        
        context_parts = []
        token_count = 0
        for i, doc in enumerate(results, 1):
            part = f"Document {i}:\n{doc.page_content}"
            if doc.metadata:
                part += f"\nSource: {doc.metadata}\n"
            part_tokens = check_token.num_tokens_from_string(part)
            if token_count + part_tokens > (MAX_TOKENS - RESERVED_FOR_ANSWER):
                break
            context_parts.append(part)
            token_count += part_tokens
        
        return "\n".join(context_parts)
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return "Error retrieving relevant context."
    
def get_enhanced_context(query, retriever):
    """Get context from RAG, static files, and CSV analysis"""
    # Get RAG context first (priority)
    rag_context = get_rag_context(query, retriever)
    
    # Get static uploaded context (non-CSV files only)
    static_context = getattr(st.session_state, 'uploaded_context', '')
    
    # Filter out CSV placeholder messages from static context
    if static_context:
        # Remove CSV placeholder lines
        static_lines = []
        for line in static_context.split('\n'):
            if not ("CSV file" in line and "ready for analysis" in line):
                static_lines.append(line)
        static_context = '\n'.join(static_lines).strip()
    
    # Get dynamic CSV analysis for current query
    csv_analysis = analyze_csv_files_dynamically(query)
    
    # Combine all contexts
    combined_parts = [f"KNOWLEDGE BASE CONTEXT:\n{rag_context}"]
    
    if static_context:
        combined_parts.append(f"UPLOADED FILES CONTEXT:\n{static_context}")
    
    if csv_analysis:
        combined_parts.append(f"DYNAMIC CSV ANALYSIS:\n{csv_analysis}")
    
    return "\n\n".join(combined_parts)