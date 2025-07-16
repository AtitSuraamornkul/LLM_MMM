import os
import time
import re
from dotenv import load_dotenv
#import doc_process
import json
import process_optimize
import process_summary
import uuid

from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv() 

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

with open("llm_input/llm_input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# initialize pinecone database
index_name = "m150-thb"  # change if desired

# check whether index exists, and create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# initialize embeddings model + vector store
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5" 
)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


optim_json_documents, optim_validation = process_optimize.process_mmm_report_complete('llm_input/llm_input.txt')
optimize_json_data = optim_json_documents

summ_json_documents, summ_validation = process_summary.process_mmm_summary_report_complete('summary_output/summary_extract_output.txt')
summary_json_data = summ_json_documents


all_json_data = optimize_json_data + summary_json_data

documents = all_json_data

# documents = [
#     Document(page_content=entry["page_content"], metadata=entry["metadata"]) for entry in all_json_data
#  ]


 # generate unique id's

uuids = [str(uuid.uuid4()) for _ in documents]

# # add to database
vector_store.add_documents(documents=documents, ids=uuids)

