1. run the mmm analysis (hitachi_mmm.ipynb) -> 

input: MMM dataset
output: output/summary_output.html   |   output/optimization_output.html

2. run the optimization extractor (optim_extract.py) -> 

input: output/optimization_output.html
output: llm_input/llm_input.txt

3. run the summary extractor (summary_extract.py) ->

input: output/summary_output.html
output: summary_output/summary_extract_output.txt


**FORMAT EXTRACTED TXT DATA INTO SUITABLE FORMAT (USE LLM OR WRITE SCRIPT TO AUTOMATE, see doc format in chroma_ingestion.py)**

INPUT EXTRACTED DATA INTO VECTOR DATABASE FOR RAG:

4. run chroma_ingestion.py to input into vector database

5. run python -m streamlit run app.py to start the app

