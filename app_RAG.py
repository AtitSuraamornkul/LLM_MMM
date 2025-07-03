
import os
import streamlit as st
from groq import Groq
import time
import re
from dotenv import load_dotenv
import llm

from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize RAG components
@st.cache_resource
def initialize_rag():
    """Initialize RAG components with caching for better performance"""
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "planidac"
    index = pc.Index(index_name)
    
    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5},
    )
    
    return retriever

def get_rag_context(query, retriever):
    """Retrieve relevant context using RAG"""
    try:
        results = retriever.invoke(query)
        
        if not results:
            return "No relevant context found in the knowledge base."
    
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"Document {i}:\n{doc.page_content}")
            if doc.metadata:
                context_parts.append(f"Source: {doc.metadata}\n")
        
        return "\n".join(context_parts)
    
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return "Error retrieving relevant context."

# Initialize RAG
retriever = initialize_rag()

# Load original optimization results (keep as fallback)
with open('llm_input/llm_input.txt', 'r') as file:
    optimization_results = file.read()

insights_report = llm.generate_llm_insights(optimization_results)

st.set_page_config(
    page_title="MMM ChatBot",
    layout="centered"
)

st.text(insights_report)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("MMM Optimization Insights ChatBot")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask me about your MMM optimization results...")

if user_prompt:
    # Add user message to chat
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Get RAG context
    with st.spinner("Retrieving relevant information..."):
        rag_context = get_rag_context(user_prompt, retriever)

    # Enhanced system prompt with RAG context
    system_prompt = f"""
    You are a business analytics assistant specializing in Marketing Mix Modeling (MMM) optimization. Your job is to answer questions and provide clear, actionable business insights for non-technical and management users.

    RELEVANT CONTEXT FROM KNOWLEDGE BASE:
    {rag_context}

    BUSINESS INSIGHTS REPORT:
    {insights_report}

    When answering user questions:
    - Prioritize information from the knowledge base context above, as it's most relevant to the user's query
    - Use the MMM optimization results and insights report as supplementary context
    - If information conflicts between sources, prioritize the knowledge base context
    - If a question asks for details, calculations, or recommendations, base your answer on the provided contexts
    - If information is not available in any context, politely state that you do not have that data
    - Avoid technical jargon and use concise, business-friendly language
    - Always cite which source your information comes from when possible

    Always remain clear, helpful, and focused on the business implications of the MMM results.
    """

    # Send message to LLM
    messages = [
        {"role": "system", "content": system_prompt},
        *st.session_state.chat_history
    ]

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )

        assistant_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display the LLM's response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            
        # Optional: Show retrieved context in an expander for transparency
        with st.expander("📚 Retrieved Context (Click to view sources)"):
            st.text(rag_context)
            
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")

# Optional: Add a sidebar with RAG settings
with st.sidebar:
    st.header("RAG Settings")
    
    # Allow users to adjust retrieval parameters
    k_docs = st.slider("Number of documents to retrieve", 1, 10, 5)
    score_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.1)
    
    if st.button("Update Retrieval Settings"):
        # Update retriever with new settings
        retriever = initialize_rag()
        retriever.search_kwargs = {"k": k_docs, "score_threshold": score_threshold}
        st.success("Settings updated!")
    
    st.info("Adjust for better responses")