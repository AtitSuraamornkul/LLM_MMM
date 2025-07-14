import os
from dotenv import load_dotenv
import json
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()

def generate_llm_insights(optimization_results):
    system_prompt = """MMM Budget Optimization Insights for Business Leaders

You are a marketing strategy advisor who specializes in turning complex Marketing Mix Modeling (MMM) optimization results into clear, actionable business insights for non-technical and management teams.

When given structured MMM optimization results (including budgets, ROI, revenue, and channel allocations before and after optimization), your job is to:

1. Summarize the key outcomes in plain language. Focus on what changed, HIGHLIGHT overall incremental revenue, what stayed the same, and what it means for the business.
2. Explain the impact of budget changes for each channel. Clearly state which channels received more or less money, and what effect this had on sales or revenue.
3. Highlight the most important takeaways. Identify which channels performed best, which ones underperformed, and where there may be opportunities for improvement.
4. Give straightforward recommendations. Offer practical suggestions for next steps, such as where to invest more or less, or what to watch going forward.
5. Point out any limitations or things to consider. Mention if there were restrictions or anything that might have limited the results, in terms that any business leader can understand.

Format your response with these headings:
- Summary
- What Changed
- Key Takeaways
- Recommendations
- Things to Consider

- CRITICAL: Replace ALL '$' (Dollar) references with 'THB' (Thai Baht) unless explicitly specified otherwise
- Format large numbers clearly: Write "2,000,000" instead of "2.0M"

Use bullet points and short paragraphs. Avoid technical jargon."""

    try:
        # Initialize Ollama LLM
        llm = ChatGroq(
            model="llama3-70b-8192",  # You can change this to your preferred model
            temperature=1,
            #base_url="http://localhost:11434",
            #num_predict=2000,  # Equivalent to max_tokens
        )

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

def generate_llm_insights_alternative(optimization_results):
    """Alternative implementation using direct string formatting"""
    system_prompt = """MMM Budget Optimization Insights for Business Leaders

You are a marketing strategy advisor who specializes in turning complex Marketing Mix Modeling (MMM) optimization results into clear, actionable business insights for non-technical and management teams.

When given structured MMM optimization results (including budgets, ROI, revenue, and channel allocations before and after optimization), your job is to:

1. Summarize the key outcomes in plain language. Focus on what changed, HIGHLIGHT overall incremental revenue, what stayed the same, and what it means for the business.
2. Explain the impact of budget changes for each channel. Clearly state which channels received more or less money, and what effect this had on sales or revenue.
3. Highlight the most important takeaways. Identify which channels performed best, which ones underperformed, and where there may be opportunities for improvement.
4. Give straightforward recommendations. Offer practical suggestions for next steps, such as where to invest more or less, or what to watch going forward.
5. Point out any limitations or things to consider. Mention if there were restrictions or anything that might have limited the results, in terms that any business leader can understand.

Format your response with these headings:
- Summary
- What Changed
- Key Takeaways
- Recommendations
- Things to Consider

Use bullet points and short paragraphs. Avoid technical jargon.

Here are our MMM optimization results:
{optimization_results}

Please provide the business insights report."""

    try:
        # Initialize Ollama LLM
        llm = ChatOllama(
            model="llama3.2:3b",
            temperature=1,
            base_url="http://localhost:11434",
        )

        # Format the complete prompt
        complete_prompt = system_prompt.format(optimization_results=optimization_results)

        # Get response
        response = llm.invoke(complete_prompt)
        
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
    # Check Ollama connection first
    is_connected, models_or_error = check_ollama_connection()
    
    if not is_connected:
        print(f"❌ Ollama connection failed: {models_or_error}")
        print("Make sure Ollama is running: 'ollama serve'")
        print("And that you have a model installed: 'ollama pull llama3.1:8b'")
        exit(1)
    
    print(f"✅ Ollama connected. Available models: {models_or_error}")
    
    # Check if preferred model is available
    preferred_model = "llama3.1:8b"
    if preferred_model not in models_or_error:
        print(f"⚠️  Preferred model '{preferred_model}' not found.")
        print(f"Available models: {models_or_error}")
        print(f"Install with: 'ollama pull {preferred_model}'")
        
        # Use first available model as fallback
        if models_or_error:
            fallback_model = models_or_error[0]
            print(f"Using fallback model: {fallback_model}")
            # You'd need to update the model in the function
    
    # Load and process the optimization results
    try:
        with open('llm_input/llm_input.txt', 'r') as file:
            content = file.read()

        optimization_results = content

        print("🔄 Generating insights with Ollama...")
        insights_report = generate_llm_insights(optimization_results)
        
        print("✅ Insights generated successfully!")
        print("\n" + "="*50)
        print(insights_report)
        print("="*50)

        # Save to file
        os.makedirs('llm_output', exist_ok=True)
        with open('llm_output/llm_OUTPUT_REPORT.txt', 'w') as f:
            f.write(insights_report)
        
        print(f"\n💾 Report saved to: llm_output/llm_OUTPUT_REPORT.txt")
        
    except FileNotFoundError:
        print("❌ Error: 'llm_input/llm_input.txt' not found")
    except Exception as e:
        print(f"❌ Error: {str(e)}")