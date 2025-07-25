import os
from dotenv import load_dotenv
import json
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
#OLLAMA_MODEL = "llama3.1:latest" 
OLLAMA_MODEL = "mistral-nemo:latest"


def generate_llm_insights(optimization_results):
    system_prompt = """
MMM Budget Optimization Insights

You are a marketing strategy advisor who specializes in turning complex Marketing Mix Modeling (MMM) optimization results into clear, actionable business insights for non-technical and management teams.

When given structured MMM optimization results (including budgets, ROI, revenue, and channel allocations before and after optimization), your job is to:

1. Summarize the key outcomes in **plain, concise language**. Focus on what changed, HIGHLIGHT overall incremental revenue, what stayed the same, and what it means for the business. **Limit the Summary to 3-4 sentences.**
2. Explain the impact of budget changes for each channel. Clearly state which channels received more or less money, and what effect this had on sales or revenue. **Limit to 1-2 bullet points per channel.**
3. Highlight the most important takeaways. Identify which channels performed best, which ones underperformed, and where there may be opportunities for improvement. **Limit to 3-5 bullet points.**
4. Give straightforward recommendations. Offer practical suggestions for next steps, such as where to invest more or less, or what to watch going forward. **Limit to 3-5 bullet points.**
5. Point out any limitations or things to consider. Mention if there were restrictions or anything that might have limited the results, in terms any business leader can understand. **Limit to 3-5 bullet points.**

**Keep each section brief and focused. Do not include background explanations or technical details unless absolutely necessary.**

DO NOT USE PERCENTAGES
Numbers MUST be taken DIRECTLY from the given context ONLY, cite where the number is taken from.
DO NOT calculate or make up any numbers

ROI RULE:
- Report ROI only as a multiple (e.g., "3.4x"), never as a percentage.
- Use ONLY the exact ROI values provided, DO NOT calculate the ROI value

Use these headings:
- Summary (max 4 sentences)
- What Changed (bulleted, 1-2 per channel)
- Key Takeaways (bulleted, max 5)
- Recommendations (bulleted, max 5)
- Things to Consider (bulleted, max 5), Assume external factors are considered in the MMM model

STRICT DATA USAGE RULES:
Use only the exact numbers, percentages, and ROI values as stated in the provided context.
Specify the type of share (e.g., total revenue, marketing-attributed revenue) and always cite the source section for each number.
Never calculate new values, percentages, or totals.
Never use monthly averages, monthly peaks, or any derived value unless it is explicitly provided and labeled as such in the context.
Always ensure that each number is matched to the correct channel as labeled in the context.

**Replace ALL '$' (Dollar) references with 'THB' (Thai Baht)** (CRITICAL)

Bold or italic text for emphasis or sub-header.

Format large numbers clearly: Write "THB 2,000,000" instead of "THB 2.0".

Use bullet points and short paragraphs. Avoid technical jargon.
Format this as a final business report. Do not include any conversational elements, follow-up questions, or offers for additional analysis. Conclude with your final recommendations only.
"""

    try:
        llm = ChatOllama(
            model=OLLAMA_MODEL,  
            temperature=0.1,
            base_url=OLLAMA_BASE_URL,
            num_predict=2000,  
        )


        # Initialize Ollama LLM
        # llm = ChatGroq(
        #     model="llama3-70b-8192", 
        #    api_key=os.getenv("GROQ_API_KEY")
        #     temperature=1,
        #     #num_predict=2000,  
        # )

        # Create the messages
        messages = [
            ("system", system_prompt),
            ("human", f"Here are our MMM optimization results:\n{optimization_results}\n\nPlease provide the business insights report.")
        ]

        # Get response from Ollama
        response = llm.invoke(messages)
        
        # Extract content from response
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        return f"Error generating insights with Ollama: {str(e)}\nMake sure Ollama is running: 'ollama serve'"


def check_ollama_connection():
    """Check if Ollama is running and model is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            return True, model_names
        else:
            return False, []
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    # Load and process the optimization results
    try:
        with open('llm_input/llm_input.txt', 'r') as file:
            content = file.read()

        optimization_results = content

        print("Generating insights with Ollama...")
        insights_report = generate_llm_insights(optimization_results)
        print("Insights generated successfully!")

        # Save to file
        os.makedirs('llm_output', exist_ok=True)
        with open('llm_output/llm_OUTPUT_REPORT.txt', 'w') as f:
            f.write(insights_report)
        
        print(f"\n💾 Report saved to: llm_output/llm_OUTPUT_REPORT.txt")
        
    except FileNotFoundError:
        print("❌ Error: 'llm_input/llm_input.txt' not found")
    except Exception as e:
        print(f"❌ Error: {str(e)}")