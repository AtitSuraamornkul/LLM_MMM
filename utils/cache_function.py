import streamlit as st
import json
import re
import pandas as pd
import os
from typing import Dict, Optional, Any

# Your existing utils
import utils.llm as llm
import utils.check_token as check_token

# RAG/Vector store imports
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize RAG components
@st.cache_resource
def initialize_rag():
    """Initialize RAG components with caching for better performance"""
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "m150-thb"
    index = pc.Index(index_name)
    
    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5},
    )
    
    return retriever


@st.cache_data
def generate_insights():
    with open('llm_input/llm_input.txt', 'r') as file:
        optimization_results = file.read()
    return llm.generate_llm_insights(optimization_results)