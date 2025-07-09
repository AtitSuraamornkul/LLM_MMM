import os
import time
import re
from dotenv import load_dotenv
import doc_process

from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# set the pinecone index

index_name = "planidac-v2"
index = pc.Index(index_name)

# initialize embeddings model + vector store

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"  # Popular lightweight model
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# retrieval

'''
###### add docs to db ##############################
results = vector_store.similarity_search_with_score(
    "what is the optimal spending for meta",
    #k=2,
    filter={"source": "marketing_analysis"},
)

print("RESULTS:")

for res in results:
    print(f"* {res[0].page_content} [{res[0].metadata}] -- {res[1]}")
'''


retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},
)
results = retriever.invoke("what is the diminishing return")

print("RESULTS:")

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")


