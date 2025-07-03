import os
import streamlit as st
from groq import Groq
import time
import re
from dotenv import load_dotenv
import llm
import check_token
import tiktoken

from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import pandas as pd
import altair as alt
import json



load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CHATBOT_AVATAR = "assets/chatbot_avatar_128x128_fixed.png"
USER_AVATAR = "assets/A3D07482-09C2-48E7-884F-EF6BABBEBFA6.PNG"


def extract_vega_dataset_from_html(html_path, chart_id):
    """
    Extracts the first dataset from the Vega-Lite spec for a given chart_id in an HTML file.
    Returns a pandas DataFrame.
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    # Find the Vega-Lite spec for the given chart_id
    pattern = re.compile(
        r'chart-embed id="' + re.escape(chart_id) + r'".*?JSON\.parse\("({.*?})"\);',
        re.DOTALL
    )
    match = pattern.search(html)
    if not match:
        st.warning(f"Could not find Vega-Lite spec for chart id '{chart_id}'")
        return None
    vega_json_str = match.group(1).encode('utf-8').decode('unicode_escape')
    vega_spec = json.loads(vega_json_str)
    # Get the dataset (first item in the 'datasets' dict)
    data = list(vega_spec['datasets'].values())[0]
    return pd.DataFrame(data)



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

MAX_TOKENS = 5000
RESERVED_FOR_ANSWER = 1000  # Reserve for LLM's answer and prompt

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

# Initialize session state for token tracking
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0

if "request_count" not in st.session_state:
    st.session_state.request_count = 0

st.title("MMM Optimization Insights ChatBot")


if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = True
    
    welcome_message = """
    🚀 **Welcome! Your MMM Analysis is Ready**
    
    I'm your AI assistant specialized in Marketing Mix Modeling insights. I've already analyzed your data and I'm ready to help you make better marketing decisions.
    
    **🎯 Popular Questions:**"

    • "Which channels should I invest more in?"
    • "What happens if I cut my TV budget by 20%?"
    • "Show me my underperforming channels"
    • "What's my ROI by channel?"
    • "Give me action items for next quarter"
    
    **📈 I can also help with:** 

    ✅ Budget reallocation strategies  
    ✅ Channel performance analysis  
    ✅ Scenario planning & forecasting  
    ✅ Implementation roadmaps  
    
    Just ask me anything about your marketing performance! 👇
    """
    
    st.session_state.chat_history.append({"role": "assistant", "content": welcome_message})

#DISPLAY HISTORY
for message in st.session_state.chat_history:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar=CHATBOT_AVATAR):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"], avatar= USER_AVATAR):
            st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask me about your MMM optimization results...")

if user_prompt:
    # Add user message to chat
    st.chat_message("user", avatar=USER_AVATAR).markdown(user_prompt)
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
    - When using the RAG context, treat \n (newline characters) as regular spacing—do not reproduce them literally in the output.
    - Clearly separate numbers and text (e.g., write "721,000 to 831,000" instead of "721,000to831,000")
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

    total_tokens = check_token.count_message_tokens(messages)
    if total_tokens > MAX_TOKENS:
        st.error(f"Prompt too long ({total_tokens} tokens, limit is {MAX_TOKENS}). Try reducing context or chat history.")
    else:
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=messages
            )

            assistant_response = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

            # Update token usage tracking
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
                st.session_state.total_tokens_used += tokens_used
            else:
                # Fallback: estimate tokens if usage not available
                estimated_tokens = check_token.count_message_tokens(messages) + check_token.num_tokens_from_string(assistant_response)
                st.session_state.total_tokens_used += estimated_tokens
            
            st.session_state.request_count += 1

            # Display the LLM's response
            with st.chat_message("assistant", avatar=CHATBOT_AVATAR):
                st.markdown(assistant_response)
                
            # Optional: Show retrieved context in an expander for transparency
            with st.expander("📚 Retrieved Context (Click to view sources)"):
                st.text(rag_context)
            
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")

# Sidebar with RAG settings and token usage
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
    
    # Token Usage Section
    st.header("Token Usage")
    
    # Display current session stats
    st.metric("Total Requests", st.session_state.request_count)
    st.metric("Total Tokens Used", f"{st.session_state.total_tokens_used:,}")
    
    if st.session_state.request_count > 0:
        avg_tokens = st.session_state.total_tokens_used / st.session_state.request_count
        st.metric("Avg Tokens per Request", f"{avg_tokens:.0f}")
    
    # Token limit progress bar
    if st.session_state.chat_history:
        current_conversation_tokens = check_token.count_message_tokens([
            {"role": "system", "content": "System prompt placeholder"},
            *st.session_state.chat_history
        ])
        progress = min(current_conversation_tokens / MAX_TOKENS, 1.0)
        st.progress(progress)
        st.caption(f"Current conversation: {current_conversation_tokens:,} / {MAX_TOKENS:,} tokens")
        
        if progress > 0.8:
            st.warning("⚠️ Approaching token limit. Consider clearing chat history.")
    
    # Reset button
    if st.button("Reset Token Counter"):
        st.session_state.total_tokens_used = 0
        st.session_state.request_count = 0
        st.success("Token counter reset!")
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")


chart_options = [
    ("Response Curves", "response-curves-chart"),
    # Add more as needed
]


tab_labels = [label for label, _ in chart_options]
tabs = st.tabs(tab_labels)

for (label, chart_id), tab in zip(chart_options, tabs):
    with tab:
        df = extract_vega_dataset_from_html('output/summary_output.html', chart_id)
        if df is not None and not df.empty:
            # Customize chart for each chart_id if needed
            # Here's a generic example for line charts with spend/mean
            if 'spend' in df.columns and 'mean' in df.columns and 'channel' in df.columns:
                chart = alt.Chart(df).mark_line().encode(
                    x=alt.X('spend', title='Spend'),
                    y=alt.Y('mean', title='Incremental outcome'),
                    color='channel:N',
                    tooltip=['channel', 'spend', 'mean']
                ).properties(
                    width=600,
                    height=400,
                    title=label
                )
                # Add current spend points if present
                if 'current_spend' in df.columns:
                    points = alt.Chart(df[df['current_spend'].notnull()]).mark_point(filled=True, size=80).encode(
                        x='spend',
                        y='mean',
                        color='channel:N',
                        shape='current_spend:N',
                        tooltip=['channel', 'spend', 'mean']
                    )
                    chart = chart + points
                st.altair_chart(chart, use_container_width=True)
            else:
                # Fallback: simple dataframe display if chart structure unknown
                st.dataframe(df)
        else:
            st.info("No data found for the selected chart.")