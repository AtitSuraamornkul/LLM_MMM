import re
import json
from langchain_core.documents import Document

def split_to_chunks(text):
    chunks = []

    # 1. Optimization Scenario Summary
    summary_match = re.search(r"Optimization Scenario Summary:(.*?)(?=\nChart Title:|\nBudget Allocation Table:|\nInsights:|\Z)", text, re.S)
    if summary_match:
        chunk = summary_match.group(1).strip()
        if chunk:
            chunks.append({'category': 'Optimization Scenario Summary', 'text': chunk})

    # 2. Budget Allocation Table
    alloc_match = re.search(r"Budget Allocation Table:(.*?)(?=\nInsights:|\nChart Title:|\Z)", text, re.S)
    if alloc_match:
        chunk = alloc_match.group(1).strip()
        if chunk:
            chunks.append({'category': 'Budget Allocation Table', 'text': chunk})

    # 3. Insights
    insights_match = re.search(r"Insights:(.*?)(?=\nChart Title:|\Z)", text, re.S)
    if insights_match:
        chunk = insights_match.group(1).strip()
        if chunk:
            chunks.append({'category': 'Insights', 'text': chunk})

    # 4. All Chart Titles and their sections
    for chart_match in re.finditer(r"Chart Title: (.*?)\nDescription:(.*?)\nData Points:(.*?)(?=\nChart Title:|\nPercentage Allocation:|\nNon-optimized Revenue:|\Z)", text, re.S):
        title = chart_match.group(1).strip()
        desc = chart_match.group(2).strip()
        data = chart_match.group(3).strip()
        full_text = f"{desc}\n{data}"
        chunks.append({'category': f'Chart: {title}', 'text': full_text})

    # 5. Percentage Allocation
    perc_match = re.search(r"Percentage Allocation:(.*?)(?=\nChart Title:|\nNon-optimized Revenue:|\Z)", text, re.S)
    if perc_match:
        chunk = perc_match.group(1).strip()
        if chunk:
            chunks.append({'category': 'Percentage Allocation', 'text': chunk})

    # 6. Optimized incremental revenue across all channels
    opt_rev_match = re.search(r"Chart Title: Optimized incremental revenue across all channels(.*?)(?=\n\*\*Channel:|\Z)", text, re.S)
    if opt_rev_match:
        chunk = opt_rev_match.group(1).strip()
        if chunk:
            chunks.append({'category': 'Optimized Incremental Revenue', 'text': chunk})

    # 7. Channel summaries
    for channel_match in re.finditer(r"\*\*Channel: ([^\*]+)\*\*(.*?)(?=\n\*\*Channel:|\Z)", text, re.S):
        channel = channel_match.group(1).strip()
        details = channel_match.group(2).strip()
        chunks.append({'category': f'Channel response curve: {channel}', 'text': details})

    return chunks


def add_document(chunks):
    documents1 = []

    for i, chunk in enumerate(chunks):
        document = Document(
            page_content=chunk['text'],
            metadata={
                "category": chunk['category'],
                "source": "marketing_analysis"
            }
        )
        documents1.append(document)

    return documents1





if __name__ == "__main__":
    with open("llm_input/llm_input.txt", "r", encoding="utf-8") as f:
         text = f.read()

    chunks = split_to_chunks(text)

    add_document(chunks)
    # with open("llm_input/llm_input.txt", "r", encoding="utf-8") as f:
    #     text = f.read()

    # chunks = split_to_chunks(text)

    # # Write to JSONL for Pinecone
    # with open("pinecone_chunks.jsonl", "w", encoding="utf-8") as f:
    #     for chunk in chunks:
    #         f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # print(f"Split into {len(chunks)} chunks. Output written to pinecone_chunks.jsonl")



    