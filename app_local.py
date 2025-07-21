import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests
import streamlit as st
import time
import re
from dotenv import load_dotenv
import utils.llm as llm
import utils.check_token as check_token
import tiktoken

import pandas as pd
import altair as alt
import json

from datetime import datetime

import utils.file_processing as file_processing
from utils.file_processing import analyze_csv_files_dynamically
import utils.cleanup as cleanup

import utils.check_llama as check_llama

import utils.cache_function as cache_function

load_dotenv()

OLLAMA_BASE_URL = "https://ollama-llama3-1--8b-776241027088.asia-southeast1.run.app"  # Default Ollama URL
OLLAMA_MODEL = "llama3.1:latest" 

CHATBOT_AVATAR = "assets/chatbot_avatar_128x128_fixed.png"
USER_AVATAR = "assets/A3D07482-09C2-48E7-884F-EF6BABBEBFA6.PNG"

def extract_vega_dataset_from_html(html_path, chart_id):
    """
    Extracts the first dataset from the Vega-Lite spec for a given chart_id in an HTML file.
    Returns a pandas DataFrame.
    """
    try:
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
        if 'datasets' not in vega_spec:
            st.warning(f"No datasets found in Vega spec for chart id '{chart_id}'")
            return None
            
        data = list(vega_spec['datasets'].values())[0]
        return pd.DataFrame(data)
    
    except Exception as e:
        st.error(f"Error extracting data for chart '{chart_id}': {str(e)}")
        return None


MAX_TOKENS = 10000
RESERVED_FOR_ANSWER = 2000  # Reserve for LLM's answer and prompt

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


retriever = cache_function.initialize_rag()

insights_report = cache_function.generate_insights()


st.set_page_config(
    page_title="MMM ChatBot",
    layout="centered"
)



cleanup.initialize_session_state()

st.title("MMM Optimization Insights ChatBot")

if not st.session_state.welcome_shown:
    st.session_state.welcome_shown = True
    
    welcome_message = """
    **Welcome! Your MMM Analysis is Ready**
    
    I'm your AI assistant specialized in Marketing Mix Modeling insights. I've already analyzed your data and I'm ready to help you make better marketing decisions.
    
    **🎯 Popular Questions:**

    • "Which channels should I invest more in?"
    • "What happens if I increase my budget by 25%?"
    • "Show me my underperforming channels"
    • "What's my ROI by channel?"
    • "Give me action items for next quarter"
    
    **📈 I can also help with:** 

    • Budget reallocation strategies  
    • Channel performance analysis  
    • Scenario planning & forecasting  
    • Implementation roadmaps  
    
    Just ask me anything about your marketing performance or click below for quick insights and dashboard view! 👇
    """
    
    st.session_state.chat_history.append({"role": "assistant", "content": welcome_message})


# DISPLAY HISTORY
for message in st.session_state.chat_history:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar=CHATBOT_AVATAR):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"], avatar=USER_AVATAR):
            st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask me about your MMM optimization results...")

if user_prompt:

    cleanup.cleanup_chat_history()

    st.session_state.current_user_prompt = user_prompt

    # Add user message to chat
    st.chat_message("user", avatar=USER_AVATAR).markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Get RAG context
    with st.spinner("Retrieving relevant information..."):
        enhanced_context = get_enhanced_context(user_prompt, retriever)


    # Enhanced system prompt with RAG context
    system_prompt = f"""
You are explaining ads to a 10-year-old kid. Make it super fun and easy!

DATA:
INFORMATION: {enhanced_context}
REPORT: {insights_report}

MUST DO:
- Change '$' to 'THB'
- Use ONLY words a 10-year-old knows (no big grown-up words!)
- Be super short - maximum 3-4 lines per point
- Use LOTS of emoji, especially faces 😊🎉✨👍👎✅❌⚠️💰📈
- Use fun comparisons: "like magic!" "like getting free candy!" "like a broken toy"
- Show clear good/bad with ✅❌ or 👍👎
- Sound excited and happy!

BANNED WORDS (Don't use these!):
- optimization, performance, incremental, revenue, investment
- analysis, suggests, prioritizing, diminishing, returns
- significant, delta, allocation, strategy

USE INSTEAD:
- "made more money" not "increased revenue"
- "works great" not "high performance" 
- "waste money" not "diminishing returns"
- "do this" not "we suggest"

EXAMPLE FORMAT:
"😊 **Great News!**
💰 You made 5 million more THB! Like finding treasure! 🎉
✅ Facebook ads work like magic! ✨
❌ TV ads are broken - they waste money 👎
🎯 **Do this:** Use more Facebook, less TV!"

Keep it SHORT, FUN, and use words a kid would say to their friend!
Only use the information given - don't make stuff up!
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
        with st.spinner("Generating Response"):
            try:
                # Prepare Ollama request
                ollama_payload = {
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False
                }
                
                # Make request to Ollama
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=ollama_payload,
                    timeout=120  # 2 minute timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    assistant_response = response_data['message']['content']
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    
                    # Update token usage tracking (estimate since Ollama doesn't provide exact counts)
                    estimated_tokens = check_token.count_message_tokens(messages) + check_token.num_tokens_from_string(assistant_response)
                    st.session_state.total_tokens_used += estimated_tokens
                    st.session_state.request_count += 1
                else:
                    st.error(f"Ollama request failed with status {response.status_code}: {response.text}")
                    assistant_response = None

                # Display the LLM's response
                with st.chat_message("assistant", avatar=CHATBOT_AVATAR):
                    st.markdown(assistant_response)
            
            # Optional: Show retrieved context in an expander for transparency
                with st.expander("📚 Retrieved Context (Click to view sources)"):
                    st.text(enhanced_context)
                    
                
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")

# Sidebar with RAG settings, file upload, and token usage
with st.sidebar:
    st.header("File Upload")
    st.caption("Supports: PDF, Word, Text files")  # Updated caption
    
    uploaded_files = st.file_uploader(
        "Upload files for additional context",
        type=['txt', 'pdf', 'docx', 'doc', 'csv'],  # Removed CSV/Excel types
        accept_multiple_files= True,  
        help="Extract content from documents"  
    )
    
    if uploaded_files:
        # Show file info
        total_size = 0
        for file in uploaded_files:
            file_size = len(file.getvalue()) / 1024  # KB
            total_size += file_size
        
        st.write(f"**Total size:** {total_size:.1f} KB")
        

        
        if st.button("Process All Files", type="primary"):
            with st.spinner("🔄 Processing files..."):
                try:
                    # Process files and separate CSV files for dynamic analysis
                    session_context = file_processing.process_uploaded_files(uploaded_files)
                    st.session_state.uploaded_context = session_context
                    
                    # Count different file types
                    csv_files = [f for f in uploaded_files if f.type == "text/csv"]
                    static_files = [f for f in uploaded_files if f.type != "text/csv"]
                    
                    st.write(f"✅ Successfully processed {len(uploaded_files)} files!")
                    if static_files:
                        st.write(f"📄 {len(static_files)} static file(s) added to context")
                    if csv_files:
                        st.write(f"📊 {len(csv_files)} CSV file(s) ready for dynamic analysis")
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
    
    # Show current uploaded context status
    if hasattr(st.session_state, 'uploaded_context') and st.session_state.uploaded_context:
        
        # Show context stats
        context_length = len(st.session_state.uploaded_context)
        estimated_tokens = context_length // 4
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Characters", f"{context_length:,}")
        with col2:
            st.metric("Est. Tokens", f"{estimated_tokens:,}")
        
        # Show context preview
        with st.expander("Preview uploaded content"):
            preview_length = min(1000, len(st.session_state.uploaded_context))
            preview = st.session_state.uploaded_context[:preview_length]
            if len(st.session_state.uploaded_context) > preview_length:
                preview += "\n\n[... more content available ...]"
            st.text(preview)
        
        if st.button("Clear All Files"):
            st.session_state.uploaded_context = ""
            st.success("Files cleared!")
            st.rerun()

    st.header("RAG Settings")
    
    # Allow users to adjust retrieval parameters
    k_docs = st.slider("Number of documents to retrieve", 1, 10, 5)
    score_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.1)
    
    if st.button("Update Retrieval Settings"):
        # Update retriever with new settings
        retriever = cache_function.initialize_rag()
        retriever.search_kwargs = {"k": k_docs, "score_threshold": score_threshold}
        st.success("Settings updated!")

    with st.expander("Check Ollama Status", expanded=False):
        st.header("Ollama Status")
    
        # Check Ollama status
        is_running, models = check_llama.check_ollama_status()
        
        if is_running:
            st.success("✅ Ollama is running")
            st.write(f"Available models: {len(models)}")
       
        else:
            st.error("❌ Ollama not running")
            st.write("Start Ollama: `ollama serve`")


    with st.expander("📊 Token Usage", expanded=False):
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

    with st.expander("Memory Management", expanded=False):
        st.subheader("Memory Management")
        
        # Show current memory usage
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chat Messages", len(st.session_state.chat_history))
            st.metric("Max Messages", st.session_state.max_chat_history)
        
        with col2:
            context_size = len(getattr(st.session_state, 'uploaded_context', ''))
            st.metric("Context Size", f"{context_size:,} chars")
            st.metric("Max Context Tokens", st.session_state.max_context_tokens)
        
        # Manual cleanup buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clean Chat History"):
                cleanup.cleanup_chat_history()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All Files"):
                st.session_state.uploaded_context = ""
                st.success("Files cleared!")
                st.rerun()



# Charts section
st.header("Marketing Mix Analysis Charts")

with st.expander("Display Charts", expanded=False):
    chart_options = [
        ("Response Curves", "response-curves-chart"),
        ("Spend and Revenue Contribution", "spend-outcome-chart"),
        # Add more as needed
    ]

    tab_labels = [label for label, _ in chart_options]
    tabs = st.tabs(tab_labels)

    for (label, chart_id), tab in zip(chart_options, tabs):
        with tab:
            df = extract_vega_dataset_from_html('output/new_summary_output.html', chart_id)
            
            if df is not None and not df.empty:     
                # Customize chart for each chart_id
                if chart_id == "response-curves-chart":
                    # Check for required columns
                    required_cols = ['spend', 'mean', 'channel']
                    if all(col in df.columns for col in required_cols):
                        chart = alt.Chart(df).mark_line(point=True, strokeWidth=1, interpolate='linear').encode(
                            x=alt.X('spend:Q', title='Spend'),
                            y=alt.Y('mean:Q', title='Incremental Outcome'),
                            color=alt.Color('channel:N', title='Channel'),
                            tooltip=['channel:N', 'spend:Q', 'mean:Q']
                        ).properties(
                            width=600,
                            height=400,
                            title=f"{label} - Response Curves by Channel"
                        )
                        
                        # Add current spend points if available
                        if 'current_spend' in df.columns:
                            current_spend_df = df[df['current_spend'].notnull()]
                            if not current_spend_df.empty:
                                points = alt.Chart(current_spend_df).mark_point(
                                    filled=True, 
                                    size=150, 
                                    shape='diamond'
                                ).encode(
                                    x='spend:Q',
                                    y='mean:Q',
                                    color='channel:N',
                                    tooltip=['channel:N', 'spend:Q', 'mean:Q', 'current_spend:N']
                                )
                                chart = chart + points
                        
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.warning(f"Missing required columns for response curves. Need: {required_cols}")
                        st.write("Available data:")
                        st.dataframe(df)
                
                elif chart_id == "spend-outcome-chart":
                    # Check for required columns
                    required_cols = ['channel', 'label', 'pct']
                    if all(col in df.columns for col in required_cols):
                        
                        # Calculate global Y-axis scale to ensure consistent scaling
                        max_pct = df['pct'].max()
                        y_scale = alt.Scale(domain=[0, max_pct * 1.1])  # Add 10% padding
                        
                        # Get unique channels
                        channels = df['channel'].unique()
                        
                        # Calculate chart width based on number of channels
                        # Limit individual chart width and total width
                        individual_chart_width = min(120, max(80, 600 // len(channels)))  # Responsive width
                        total_width = individual_chart_width * len(channels)/2
                        
                        # Create individual charts for each channel
                        charts = []
                        for i, channel in enumerate(channels):
                            channel_df = df[df['channel'] == channel]
                            
                            # Bar chart for this channel
                            bars = alt.Chart(channel_df).mark_bar(
                                cornerRadiusTopLeft=2,
                                cornerRadiusTopRight=2
                            ).encode(
                                x=alt.X('label:N', 
                                    title=None, 
                                    axis=alt.Axis(labelAngle=0, labels=False, ticks=False),
                                    scale=alt.Scale(paddingOuter=0.5)),
                                y=alt.Y('pct:Q', 
                                    title='%' if i == 0 else None,  # Only show Y-axis title on first chart
                                    axis=alt.Axis(format='%', tickCount=5) if i == 0 else alt.Axis(format='%', tickCount=5, labels=False),
                                    scale=y_scale),  # Use consistent scale
                                color=alt.Color('label:N',
                                    scale=alt.Scale(
                                        domain=['% Revenue', '% Spend'],
                                        range=['#669DF6', '#AECBFA']  # Match your original colors
                                    ),
                                    legend=None  # Remove individual legends
                                ),
                                tooltip=['channel:N', 'label:N', alt.Tooltip('pct:Q', format='.1%')]
                            ).properties(
                                width=individual_chart_width,  # Use calculated width
                                height=300,
                                title=alt.TitleParams(
                                    text=channel,
                                    fontSize=12,
                                    anchor='start',
                                    color='#3C4043'
                                )
                            )
                            
                            # Add ROI indicators if available
                            if 'roi' in df.columns:
                                # Get ROI value for this channel
                                roi_value = channel_df['roi'].iloc[0] if len(channel_df) > 0 else None
                                
                                if roi_value is not None:
                                    # ROI tick mark
                                    roi_tick = alt.Chart(pd.DataFrame([{
                                        'x_pos': 0.5,  # Center position
                                        'y_pos': max_pct * 0.85,  # Position near top
                                        'roi': roi_value
                                    }])).mark_tick(
                                        color='#188038',
                                        thickness=4,
                                        size=30,
                                        orient='horizontal'
                                    ).encode(
                                        x=alt.X('x_pos:Q', scale=alt.Scale(domain=[0, 1]), axis=None),
                                        y=alt.Y('y_pos:Q', scale=y_scale, axis=None)
                                    )
                                    
                                    # ROI text label
                                    roi_text = alt.Chart(pd.DataFrame([{
                                        'x_pos': 0.5,
                                        'y_pos': max_pct * 0.9,  # Slightly above tick
                                        'roi': roi_value
                                    }])).mark_text(
                                        color='#202124',
                                        fontSize=12,
                                        fontWeight='normal'
                                    ).encode(
                                        x=alt.X('x_pos:Q', scale=alt.Scale(domain=[0, 1]), axis=None),
                                        y=alt.Y('y_pos:Q', scale=y_scale, axis=None),
                                        text=alt.Text('roi:Q', format='.1f')
                                    )
                                    
                                    bars = bars + roi_tick + roi_text
                                
                                # Add ROI to tooltip
                                bars = bars.encode(
                                    tooltip=['channel:N', 'label:N', alt.Tooltip('pct:Q', format='.1%'), alt.Tooltip('roi:Q', format='.2f', title='ROI')]
                                )
                            
                            charts.append(bars)
                        
                        # Concatenate all charts horizontally with shared Y-axis
                        if len(charts) > 1:
                            combined_chart = alt.hconcat(*charts).resolve_scale(
                                y='shared'  # Use shared Y-axis for proper comparison
                            )
                        else:
                            combined_chart = charts[0]
                        
                        # Add overall title
                        final_chart = combined_chart.properties(
                            title=alt.TitleParams(
                                text="Spend and revenue contribution by marketing channel",
                                fontSize=18,
                                anchor='start',
                                color='#3C4043',
                                fontWeight='normal'
                            )
                        )
                        
                        if total_width > 200:  # If chart is too wide
                            st.markdown('<div style="width:100%; overflow-x:auto;">', unsafe_allow_html=True)
                            st.altair_chart(final_chart, use_container_width=False)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.caption("💡 Scroll horizontally to view all channels →")
                        else:
                            st.altair_chart(final_chart, use_container_width=True)
                            
                        
                        # Add legend manually below the chart
                        st.markdown("""
                        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px;">
                            <div style="display: flex; align-items: center; gap: 5px;">
                                <div style="width: 16px; height: 16px; background-color: #669DF6; border-radius: 2px;"></div>
                                <span style="font-size: 12px; color: #5F6368;">% Revenue</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 5px;">
                                <div style="width: 16px; height: 16px; background-color: #AECBFA; border-radius: 2px;"></div>
                                <span style="font-size: 12px; color: #5F6368;">% Spend</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 5px;">
                                <div style="width: 16px; height: 4px; background-color: #188038; border-radius: 2px;"></div>
                                <span style="font-size: 12px; color: #5F6368;">Return on Investment</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add note
                        st.caption("Note: Return on investment is calculated by dividing the revenue attributed to a channel by marketing costs.")

if st.button("Click For Budget Optimization Summary Report"):
    st.markdown(insights_report)

if st.button("Click For Optimization Dashboard"):
    try:
        with open('output/new_optimization_output.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Store in session state to trigger modal
        st.session_state.show_modal = True
        st.session_state.html_content = html_content
        
    except FileNotFoundError:
        st.error("Optimization report file not found at 'output/optimization_output.html'")
    except Exception as e:
        st.error(f"Error loading optimization report: {str(e)}")

# Modal dialog
if st.session_state.get("show_modal", False):
    @st.dialog("Optimization Report", width="large")
    def show_html_modal():
        # Override to make it larger
        st.markdown("""
        <style>
        .stDialog > div[data-testid="modal"] {
            width: 95vw !important;
            max-width: 95vw !important;
            height: 90vh !important;
        }
        .stDialog > div[data-testid="modal"] > div {
            width: 100% !important;
            height: 100% !important;
            max-width: 100% !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.components.v1.html(
            st.session_state.html_content, 
            height=900,  # Increased height
            scrolling=True
        )

        st.session_state.show_modal = False
        
    
    show_html_modal()