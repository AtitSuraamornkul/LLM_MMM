import os
import time
import re
from dotenv import load_dotenv
import doc_process
import json

from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv() 

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

with open("llm_input/llm_input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# initialize pinecone database
index_name = "planidac"  # change if desired

# check whether index exists, and create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# initialize embeddings model + vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Popular lightweight model
)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


#chunks = doc_process.split_to_chunks(text)
#documents =  doc_process.add_document(chunks)

# Read and process JSON ---
with open("marketing_analysis_rag_documents.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

documents = [
    Document(page_content=entry["page_content"], metadata=entry["metadata"])
    for entry in json_data
]


 # generate unique id's

start_id = 11
i = 0
uuids = []

while i < len(documents):
    i += 1
    uuids.append(f"id{start_id+i}")

# # add to database
vector_store.add_documents(documents=documents, ids=uuids)