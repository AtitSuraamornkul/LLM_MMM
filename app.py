import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import llm
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

with open('llm_input/llm_input.txt', 'r') as file:
    optimization_results = file.read()

insights_report = llm.generate_llm_insights(optimization_results)

st.set_page_config(
    page_title="MMM ChatBot",
    layout="centered"
)

st.text(
  insights_report
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("MMM Optimization Insights ChatBot")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask me about your MMM optimization results...")

# System prompt
system_prompt = f"""
You are a business analytics assistant specializing in Marketing Mix Modeling (MMM) optimization. Your job is to answer questions and provide clear, actionable business insights for non-technical and management users.

Here are the latest MMM optimization results:
{optimization_results}

Here is the business insights report based on these results:
{insights_report}

When answering user questions:
- Always use the information above as your primary context.
- If a question asks for details, calculations, or recommendations, base your answer strictly on the provided results and report.
- If information is not available in the context, politely state that you do not have that data.
- Avoid technical jargon and use concise, business-friendly language.

If the user asks for clarification, further analysis, or scenario planning, use your expertise to guide them using only the information above.

Always remain clear, helpful, and focused on the business implications of the MMM results.
"""

if user_prompt:
    # Add user message to chat
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Send user's message to the LLM and get a response
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
            
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")

