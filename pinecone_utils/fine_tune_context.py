import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Setup
load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "m150-thb"
index = pc.Index(index_name)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},
)

# List of questions
questions = [
    "What is the percentage increase in incremental revenue after optimization?",
    "Which marketing channel saw the largest absolute increase in optimized spend, and by how much?",
    "What is the difference in ROI between the best and worst performing channels?",
    "Which marketing channel is the most efficient in terms of ROI and why?",
    "Which channels are underperforming and should have their budgets reduced?",
    "How does Facebook’s performance compare to YouTube in terms of ROI and revenue contribution?",
    "If you had to cut 10% from the overall budget, which channels would you reduce first and why?",
    "Which channels should receive increased investment, and what is the expected impact?",
    "What proportion of total revenue is attributable to paid marketing vs. baseline (organic)?",
    "For the Activation channel, was a point of diminishing returns identified? What does this mean for future budget allocation?",
    "Are there any channels where the difference in ROI is not statistically significant?",
    "How is ROI calculated in this analysis?",
    "What does marginal ROI mean, and why is it important for budget decisions?",
    "What would happen to total incremental revenue if the Activation budget was increased by 25%?",
    "How did TV_Spot’s performance change after optimization?",
    "What is the current spend and revenue contribution for each marketing channel?",
    "Which channel has the highest marginal ROI, and what does this imply for future investment?",
    "How reliable are the ROI estimates for each channel based on their confidence intervals?",
    "Which channels are showing clear signs of diminishing returns?",
    "What is the optimal spend range for the KOL channel, and what is the expected revenue at that level?",
    "How does the performance of Radio compare to TV_Spot in terms of ROI and incremental revenue?",
    "What is the impact on overall ROI if budget is reallocated from underperforming to high-performing channels?",
    "Which channels contributed the most to the incremental revenue gain after optimization?",
    "What is the average effectiveness score for each channel, and which is the most effective?",
    "How does the cost per incremental KPI (CPIK) vary across channels?",
    "Which channels have the widest and narrowest ROI confidence intervals, and what does this mean?",
    "How does the optimized budget allocation differ from the non-optimized allocation for each channel?",
    "What is the business implication of the high baseline (organic) revenue share in total revenue?",
    "How do scenario analyses (e.g., +25% spend on YouTube) affect incremental revenue and ROI?",
    "What are the key actionable recommendations for the next marketing budget cycle based on these results?"
]

output_path = "rag_mmm_context_questions.jsonl"
with open(output_path, "w", encoding="utf-8") as fout:
    for idx, q in enumerate(questions, 1):
        results = retriever.invoke(q)
        results = results[:5]
        context_blocks = "\n\n".join(res.page_content for res in results)
        prompt = f"CONTEXT:\n{context_blocks}\nQ: {q}\nA:"
        fout.write(json.dumps({"prompt": prompt, "completion": ""}, ensure_ascii=False) + "\n")
        print(f"[{idx}/{len(questions)}] Finished: {q}")

print(f"\nSaved {len(questions)} prompt-context pairs (5 docs each) to {output_path}")