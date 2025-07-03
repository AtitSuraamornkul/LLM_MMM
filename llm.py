import os
from dotenv import load_dotenv
import groq
import json
import pandas as pd 

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = groq.Client(api_key=api_key)

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

Use bullet points and short paragraphs. Avoid technical jargon."""

    # Convert results to JSON string for the prompt
    results_str = json.dumps(optimization_results, indent=2)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Here are our MMM optimization results:\n{results_str}\n\nPlease provide the business insights report."
            }
        ],
        model="llama3-70b-8192",  
        temperature=1, 
        max_tokens=2000
    )
    
    return chat_completion.choices[0].message.content
  
if __name__ == "__main__":
    with open('llm_input/llm_input.txt', 'r') as file:
            content = file.read()

    optimization_results = content

    insights_report = generate_llm_insights(optimization_results)
    print(insights_report)

    with open('llm_output/llm_OUTPUT_REPORT.txt', 'w') as f:
        f.write(insights_report)




