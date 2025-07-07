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

# --- SQL AGENT INTEGRATION START ---
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.llms import Ollama
import sqlite3
from typing import List, Dict, Tuple, Optional
# --- SQL AGENT INTEGRATION END ---

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

# Enhanced SQL functions
def preprocess_column_names(df: pd.DataFrame) -> Dict[str, str]:
    """Create a mapping of normalized column names to actual column names"""
    column_mapping = {}
    
    for col in df.columns:
        # Original column name
        column_mapping[col.lower()] = col
        
        # Remove special characters and spaces
        normalized = re.sub(r'[^a-zA-Z0-9]', '', col.lower())
        column_mapping[normalized] = col
        
        # Split on common separators and add individual words
        words = re.split(r'[_\s\-\.]+', col.lower())
        for word in words:
            if len(word) > 2:  # Only add words longer than 2 characters
                column_mapping[word] = col
    
    return column_mapping

def get_sample_values_for_columns(df: pd.DataFrame, max_samples: int = 5) -> Dict[str, List]:
    """Get sample values for each column to help with query generation"""
    sample_values = {}
    
    for col in df.columns:
        try:
            # Get unique non-null values
            unique_vals = df[col].dropna().unique()
            
            # For numeric columns, show min/max/avg
            if df[col].dtype in ['int64', 'float64']:
                sample_values[col] = {
                    'type': 'numeric',
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'avg': float(df[col].mean()),
                    'samples': [str(x) for x in unique_vals[:max_samples]]
                }
            else:
                sample_values[col] = {
                    'type': 'categorical',
                    'unique_count': len(unique_vals),
                    'samples': [str(x) for x in unique_vals[:max_samples]]
                }
        except Exception as e:
            sample_values[col] = {
                'type': 'unknown',
                'error': str(e),
                'samples': []
            }
    
    return sample_values

def detect_query_intent(question: str) -> Dict[str, any]:
    """Analyze the question to understand what type of query is needed"""
    question_lower = question.lower()
    
    intent = {
        'aggregation': False,
        'grouping': False,
        'filtering': False,
        'sorting': False,
        'comparison': False,
        'top_n': False,
        'time_series': False,
        'keywords': []
    }
    
    # Aggregation patterns
    agg_patterns = [
        (r'\b(total|sum|count|average|avg|mean|max|maximum|min|minimum)\b', 'aggregation'),
        (r'\b(group by|by channel|by campaign|per channel|per campaign)\b', 'grouping'),
        (r'\b(top|bottom|best|worst|highest|lowest)\s+(\d+)', 'top_n'),
        (r'\b(where|filter|only|specific|particular)\b', 'filtering'),
        (r'\b(sort|order|arrange)\b', 'sorting'),
        (r'\b(compare|vs|versus|against)\b', 'comparison'),
        (r'\b(over time|by date|by month|by week|trend)\b', 'time_series')
    ]
    
    for pattern, intent_type in agg_patterns:
        if re.search(pattern, question_lower):
            intent[intent_type] = True
            
    # Extract potential filter values
    quoted_values = re.findall(r'"([^"]*)"', question)
    intent['quoted_values'] = quoted_values
    
    # Extract numbers
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)
    intent['numbers'] = [float(x) for x in numbers]
    
    return intent

def generate_enhanced_sql_prompt(question: str, df: pd.DataFrame, sample_values: Dict, column_mapping: Dict) -> str:
    """Generate a comprehensive prompt for SQL generation with better context"""
    
    # Analyze query intent
    intent = detect_query_intent(question)
    
    # Create detailed schema with sample values
    schema_details = []
    for col in df.columns:
        col_info = sample_values.get(col, {})
        col_type = str(df[col].dtype)
        
        if col_info.get('type') == 'numeric':
            detail = f"- {col} ({col_type}) - NUMERIC: min={col_info['min']:.2f}, max={col_info['max']:.2f}, avg={col_info['avg']:.2f}"
        elif col_info.get('type') == 'categorical':
            samples = col_info.get('samples', [])
            detail = f"- {col} ({col_type}) - CATEGORICAL: {col_info['unique_count']} unique values, examples: {', '.join(samples)}"
        else:
            samples = col_info.get('samples', [])
            detail = f"- {col} ({col_type}) - examples: {', '.join(samples[:3])}"
        
        schema_details.append(detail)
    
    # Show sample data more contextually
    sample_data_str = df.head(3).to_string(max_cols=8)
    
    # Intent-specific guidance
    intent_guidance = ""
    if intent['aggregation']:
        intent_guidance += "- This query needs aggregation (SUM, COUNT, AVG, etc.)\n"
    if intent['grouping']:
        intent_guidance += "- This query needs GROUP BY clause\n"
    if intent['top_n']:
        intent_guidance += "- This query needs ORDER BY with LIMIT\n"
    if intent['filtering']:
        intent_guidance += "- This query needs WHERE clause for filtering\n"
    
    prompt = f"""
    You are an expert SQL query generator for marketing analytics data.
    
    USER QUESTION: {question}
    
    DATABASE SCHEMA - Table: marketing_data
    EXACT Column Names (case-sensitive, use these EXACTLY as shown):
    {chr(10).join(schema_details)}
    
    SAMPLE DATA:
    {sample_data_str}
    
    QUERY ANALYSIS:
    {intent_guidance}
    
    CRITICAL REQUIREMENTS:
    1. Use ONLY the exact column names listed above - DO NOT modify them
    2. All column names are case-sensitive and must match exactly
    3. Use proper SQLite syntax
    4. If unsure about column names, use the closest match from the list above
    5. Return ONLY the SQL query - no explanations, no markdown, no extra text
    6. Don't use semicolons at the end
    7. For marketing questions, common patterns:
       - Revenue analysis: Focus on revenue, spend, ROI columns
       - Channel analysis: Group by channel-related columns
       - Campaign analysis: Group by campaign-related columns
       - Performance: Use ORDER BY for ranking
    
    Generate the SQL query:
    """
    
    return prompt

def validate_and_fix_sql(sql_query: str, available_columns: List[str]) -> Tuple[str, List[str]]:
    """Validate SQL query and attempt to fix common issues"""
    issues = []
    fixed_query = sql_query
    
    # Remove common markdown artifacts
    fixed_query = re.sub(r'```sql\s*', '', fixed_query)
    fixed_query = re.sub(r'```\s*', '', fixed_query)
    fixed_query = fixed_query.strip()
    
    # Remove trailing semicolon
    fixed_query = fixed_query.rstrip(';')
    
    # Check for column names that might not exist
    column_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
    potential_columns = re.findall(column_pattern, fixed_query)
    
    # Create case-insensitive mapping
    column_mapping = {col.lower(): col for col in available_columns}
    
    for col in potential_columns:
        if col.upper() not in ['SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER', 'HAVING', 'LIMIT', 'AS', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'IS', 'NULL', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'DESC', 'ASC', 'DISTINCT']:
            if col not in available_columns:
                # Try to find a close match
                col_lower = col.lower()
                if col_lower in column_mapping:
                    fixed_query = fixed_query.replace(col, column_mapping[col_lower])
                    issues.append(f"Fixed column name: {col} -> {column_mapping[col_lower]}")
                else:
                    # Try fuzzy matching
                    best_match = None
                    best_score = 0
                    for available_col in available_columns:
                        if col_lower in available_col.lower() or available_col.lower() in col_lower:
                            score = len(set(col_lower) & set(available_col.lower()))
                            if score > best_score:
                                best_score = score
                                best_match = available_col
                    
                    if best_match and best_score > 2:
                        fixed_query = fixed_query.replace(col, best_match)
                        issues.append(f"Fuzzy matched column: {col} -> {best_match}")
                    else:
                        issues.append(f"Warning: Column '{col}' not found in available columns")
    
    return fixed_query, issues

def generate_sql_from_natural_language_enhanced(question: str, df: pd.DataFrame, client) -> Dict[str, any]:
    """Enhanced version of SQL generation with better error handling and validation"""
    try:
        # Preprocess data
        sample_values = get_sample_values_for_columns(df)
        column_mapping = preprocess_column_names(df)
        
        # Generate enhanced prompt
        prompt = generate_enhanced_sql_prompt(question, df, sample_values, column_mapping)
        
        # Call LLM
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Validate and fix the query
        fixed_query, issues = validate_and_fix_sql(sql_query, df.columns.tolist())
        
        return {
            'sql_query': fixed_query,
            'original_query': sql_query,
            'issues': issues,
            'success': True,
            'column_mapping': column_mapping
        }
        
    except Exception as e:
        return {
            'sql_query': None,
            'original_query': None,
            'issues': [f"Error generating SQL: {str(e)}"],
            'success': False,
            'error': str(e)
        }

@st.cache_resource
def setup_database():
    """Enhanced database setup with better error handling and metadata"""
    try:
        # Load the dataset
        df = pd.read_csv("dataset/Hitachi_dataset - FULL_merged_output (5).csv")
        
        # Clean column names (remove extra spaces, special characters)
        df.columns = df.columns.str.strip()
        
        # Create SQLite connection
        conn = sqlite3.connect("marketing.db", check_same_thread=False)
        
        # Store data in SQLite
        df.to_sql("marketing_data", conn, if_exists="replace", index=False)
        
        # Create indexes for common query patterns
        common_columns = ['channel', 'campaign', 'date']  # Adjust based on your actual columns
        for col in common_columns:
            if col in df.columns:
                try:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON marketing_data({col})")
                except:
                    pass  # Index creation failed, continue
        
        conn.commit()
        conn.close()
        
        # Get enhanced metadata
        sample_values = get_sample_values_for_columns(df)
        column_mapping = preprocess_column_names(df)
        
        return df.columns.tolist(), df.head(10), df.dtypes.to_dict(), sample_values, column_mapping
    
    except Exception as e:
        st.error(f"Error setting up database: {str(e)}")
        return [], pd.DataFrame(), {}, {}, {}
    
def execute_sql_query(query, max_rows=100):
    try:
        conn = sqlite3.connect("marketing.db")
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Limit rows for display
        if len(df) > max_rows:
            df = df.head(max_rows)
            
        return df, None

    except Exception as e:
        return None, str(e)
    
def is_sql_query(user_input):
    """Check if user input contains SQL-like keywords"""
    sql_keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
    user_input_upper = user_input.upper()
    return any(keyword in user_input_upper for keyword in sql_keywords)

def is_sql_query_enhanced(user_input: str) -> bool:
    """Enhanced SQL query detection with better patterns"""
    user_input_lower = user_input.lower()
    
    # Direct SQL keywords
    sql_keywords = ['select', 'from', 'where', 'group by', 'order by', 'having', 'limit']
    if any(keyword in user_input_lower for keyword in sql_keywords):
        return True
    
    # Data analysis patterns
    analysis_patterns = [
        r'\b(show|display|get|find|list|give me)\s+(data|records|rows)',
        r'\b(top|bottom|best|worst|highest|lowest)\s+\d+',
        r'\b(total|sum|count|average|avg|mean|max|min)\s+\w+',
        r'\b(group by|by|per)\s+\w+',
        r'\b(compare|vs|versus|against)',
        r'\b(performance|revenue|spend|cost|roi)\s+(by|per|for)',
        r'\b(campaign|channel|source|medium)\s+(analysis|performance|data)',
        r'\b(how much|how many|what is the|which)',
        r'\b(breakdown|analysis|report|summary)',
        r'\b(filter|where|only|specific|particular)'
    ]
    
    return any(re.search(pattern, user_input_lower) for pattern in analysis_patterns)

def get_data_insights(df, original_question):
    if df.empty:
        return "No data found for your query."
    
    # Create a summary of the data
    summary = f"""
    Data Summary:
    - Rows returned: {len(df)}
    - Columns: {', '.join(df.columns.tolist())}
    """
    
    # Add numeric summaries if available
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            summary += f"\n- {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, avg={df[col].mean():.2f}"
    
    insight_prompt = f"""
    Based on this marketing data query result, provide 2-3 key business insights in bullet points.
    
    Original Question: {original_question}
    
    {summary}
    
    Provide actionable insights for marketing decisions:
    """
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": insight_prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except:
        return "• Data retrieved successfully\n• Use the analysis above to make informed decisions\n• Consider the patterns and trends shown in the results"

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

MAX_TOKENS = 6000
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

# Initialize database with enhanced setup
available_columns, sample_data, data_types, sample_values, column_mapping = setup_database()

# Load original optimization results (keep as fallback)
with open('llm_input/llm_input.txt', 'r') as file:
    optimization_results = file.read()

insights_report = llm.generate_llm_insights(optimization_results)

st.set_page_config(
    page_title="MMM ChatBot",
    layout="centered"
)

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
    
    **🎯 Popular Questions:**

    • "Which channels should I invest more in?"
    • "What happens if I cut my budget by 20%?"
    • "Show me my underperforming channels"
    • "What's my ROI by channel?"
    • "Give me action items for next quarter"
    
    **📊 Data Analysis Questions:**
    
    • "Show me top 10 campaigns by revenue"
    • "What's the average spend by channel?"
    • "Which campaigns have ROI above 3.0?"
    • "Compare performance across channels"
    
    **📈 I can also help with:** 

    ✅ Budget reallocation strategies  
    ✅ Channel performance analysis  
    ✅ Scenario planning & forecasting  
    ✅ Implementation roadmaps  
    ✅ Real-time data insights
    
    Just ask me anything about your marketing performance! 👇
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
    # Add user message to chat
    st.chat_message("user", avatar=USER_AVATAR).markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Check if this is a SQL-related query using enhanced detection
    is_sql_request = (
        is_sql_query_enhanced(user_prompt) or 
        any(word in user_prompt.lower() for word in [
            'show me data', 'query', 'database', 'table', 'analyze data',
            'top campaigns', 'performance by', 'average spend', 'total revenue',
            'group by', 'filter', 'sort by', 'count', 'sum', 'data from',
            'which campaigns', 'what channels', 'how much', 'compare',
            'show me', 'list', 'find', 'get data', 'breakdown'
        ])
    )
    
    if is_sql_request and available_columns:
        # Handle SQL queries using enhanced Groq
        with st.spinner("🤖 Analyzing your data with enhanced AI..."):
            
            # Check if user provided direct SQL or needs generation
            if is_sql_query(user_prompt):
                # User provided SQL directly
                sql_query = user_prompt.strip()
                st.code(sql_query, language="sql")
            else:
                # Generate SQL using enhanced method
                result = generate_sql_from_natural_language_enhanced(
                    user_prompt, sample_data, client
                )
                
                if result['success']:
                    sql_query = result['sql_query']
                    
                    # Show the generated SQL with improvements
                    st.code(sql_query, language="sql")
                    
                    if result['issues']:
                        with st.expander("🔧 Query Improvements Made"):
                            for issue in result['issues']:
                                st.info(f"• {issue}")
                else:
                    st.error(f"❌ Could not generate SQL: {result['error']}")
                    sql_query = None
            
            if sql_query:
                # Execute the query
                result_df, error = execute_sql_query(sql_query)
                
                if error:
                    # Display error message in natural language
                    error_response = f"""
I encountered an issue while analyzing your data. The query couldn't be executed due to: {error}

Let me help you with your marketing question in a different way. You can ask me about:
- Channel performance and ROI
- Campaign effectiveness 
- Budget allocation recommendations
- Marketing mix optimization

Could you rephrase your question? I have access to data about: {', '.join(available_columns[:5])}{'...' if len(available_columns) > 5 else ''}
"""
                    
                    with st.chat_message("assistant", avatar=CHATBOT_AVATAR):
                        st.markdown(error_response)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": error_response})
                    
                else:
                    # Get RAG context for additional insights
                    rag_context = get_rag_context(user_prompt, retriever)
                    
                    # Combine SQL results with RAG insights using Groq
                    combined_prompt = f"""
                    You are a marketing analytics expert. A user asked: "{user_prompt}"
                    
                    I've retrieved the following data from their marketing database:
                    
                    Query Results Summary:
                    - Total rows found: {len(result_df)}
                    - Data columns: {', '.join(result_df.columns.tolist())}
                    
                    Key data points from the results:
                    {result_df.head(10).to_string()}
                    
                    Additional Context from Knowledge Base:
                    {rag_context}
                    
                    Provide a comprehensive business response in natural language that:
                    1. Directly answers their question using the data
                    2. Explains what the numbers mean in business terms
                    3. Provides specific, actionable recommendations
                    4. Connects findings to broader marketing strategy
                    5. Uses conversational, business-friendly language
                    6. Focus entirely on business insights and next steps
                    7. Include specific numbers and percentages from the data
                    
                    Write as if you're a marketing consultant explaining findings to a client.
                    """
                    
                    try:
                        combined_response = client.chat.completions.create(
                            model="meta-llama/llama-4-maverick-17b-128e-instruct",
                            messages=[{"role": "user", "content": combined_prompt}],
                            max_tokens=800,
                            temperature=0.3
                        )
                        
                        assistant_response = combined_response.choices[0].message.content.strip()
                        
                        # Update token tracking
                        if hasattr(combined_response, 'usage') and combined_response.usage:
                            st.session_state.total_tokens_used += combined_response.usage.total_tokens
                        
                    except Exception as e:
                        # Fallback response using basic data insights
                        assistant_response = get_data_insights(result_df, user_prompt)
                    
                    # Display natural language response only
                    with st.chat_message("assistant", avatar=CHATBOT_AVATAR):
                        st.markdown(assistant_response)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    st.session_state.request_count += 1
                    
    else:
        # Original RAG-based response for non-SQL queries
        with st.spinner("Retrieving relevant information..."):
            rag_context = get_rag_context(user_prompt, retriever)

        # Enhanced system prompt with RAG context AND database info
        system_prompt = f"""
        You are a business analytics assistant specializing in Marketing Mix Modeling (MMM) optimization. Your job is to answer questions and provide clear, actionable business insights for non-technical and management users.

        AVAILABLE DATABASE:
        You have access to a marketing database with the following columns: {', '.join(available_columns) if available_columns else 'Database not available'}
        
        If users ask for specific data analysis, you can access and analyze their actual marketing data to provide precise answers.

        RELEVANT CONTEXT FROM KNOWLEDGE BASE:
        {rag_context}
        
        BUSINESS INSIGHTS REPORT:
        {insights_report}

        When answering user questions:
        - **IMPORTANT** When using the RAG context, treat \\n (newline characters) and other symbols e.g.==, | as regular spacing—do not reproduce them literally in the output.
        - Clearly separate numbers and text (e.g., write "721,000 to 831,000" instead of "721,000to831,000")
        - Prioritize information from the knowledge base context above, as it's most relevant to the user's query
        - Use the MMM optimization results and insights report as supplementary context
        - If information conflicts between sources, prioritize the knowledge base context
        - If a question asks for details, calculations, or recommendations, base your answer on the provided contexts
        - If information is not available in any context, politely state that you do not have that data
        - Avoid technical jargon and use concise, business-friendly language
        - Always cite which source your information comes from when possible
        - If users need specific data analysis, let them know you can analyze their actual data

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
    
    # Database Debug Section
    if available_columns:
        st.header("🔍 Database Info")
        
        with st.expander("Database Schema"):
            st.write("**Available Columns:**")
            for i, col in enumerate(available_columns):
                col_type = data_types.get(col, 'unknown')
                st.write(f"{i+1}. `{col}` ({col_type})")
        
        with st.expander("Sample Data"):
            st.dataframe(sample_data)
        
        # Quick test query
        if st.button("🧪 Test Database"):
            test_query = "SELECT COUNT(*) as total_rows FROM marketing_data"
            result_df, error = execute_sql_query(test_query)
            if error:
                st.error(f"Connection failed: {error}")
            else:
                st.success(f"✅ Database connected! Total rows: {result_df.iloc[0, 0]}")
    
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

# Charts section (keep your existing chart code)
st.header("📊 Marketing Mix Analysis Charts")

chart_options = [
    ("Response Curves", "response-curves-chart"),
    ("Spend and Revenue Contribution", "spend-outcome-chart"),
]

tab_labels = [label for label, _ in chart_options]
tabs = st.tabs(tab_labels)

for (label, chart_id), tab in zip(chart_options, tabs):
    with tab:
        df = extract_vega_dataset_from_html('output/summary_output.html', chart_id)
        
        if df is not None and not df.empty:     
            # Your existing chart code here...
            if chart_id == "response-curves-chart":
                # Keep your existing response curves chart code
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
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning(f"Missing required columns for response curves. Need: {required_cols}")
                    st.dataframe(df)
            
            elif chart_id == "spend-outcome-chart":
                # Keep your existing spend-outcome chart code
                required_cols = ['channel', 'label', 'pct']
                if all(col in df.columns for col in required_cols):
                    # Your existing chart code here
                    st.info("Spend and Revenue chart would be displayed here")
                    st.dataframe(df)
                else:
                    st.warning(f"Missing required columns. Need: {required_cols}")
                    st.dataframe(df)
        else:
            st.info(f"No data found for '{label}' chart.")

if st.button("Summary Report"):
    st.text(insights_report)

if st.button("View Optimization Dashboard"):
    try:
        with open('output/optimization_output.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.session_state.show_modal = True
        st.session_state.html_content = html_content
        
    except FileNotFoundError:
        st.error("Optimization report file not found at 'output/optimization_output.html'")
    except Exception as e:
        st.error(f"Error loading optimization report: {str(e)}")

# Modal dialog (keep your existing modal code)
if st.session_state.get("show_modal", False):
    @st.dialog("Optimization Report", width="large")
    def show_html_modal():
        st.markdown("""
        <style>
        .stDialog > div[data-testid="modal"] {
            width: 95vw !important;
            max-width: 95vw !important;
            height: 90vh !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.components.v1.html(
            st.session_state.html_content, 
            height=900,
            scrolling=True
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.download_button(
                label="📄 Download Report",
                data=st.session_state.html_content,
                file_name="optimization_report.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col3:
            if st.button("❌ Close", use_container_width=True):
                st.session_state.show_modal = False
                st.rerun()
    
    show_html_modal()