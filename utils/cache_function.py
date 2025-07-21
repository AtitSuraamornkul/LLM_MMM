import streamlit as st
import json
import re
import pandas as pd
import os
from typing import Dict, Optional, Any

# Your existing utils
import utils.llm as llm
import utils.check_token as check_token

# RAG/Vector store imports - CHANGED
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize RAG components
@st.cache_resource
def initialize_rag():
    """Initialize RAG components with caching for better performance"""
    # Initialize ChromaDB - CHANGED
    persist_directory = "./chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = "new-m150-thb"
    
    # Initialize embeddings and vector store - CHANGED
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )
    
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    # Create retriever - SAME
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