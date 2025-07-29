1. run the mmm analysis (hitachi_mmm.ipynb) -> 

input: MMM dataset
output: output/summary_output.html   |   output/optimization_output.html

2. run the extractor (html_extractor/all_extractor.py) -> 

input:  output/optimization_output.html
        output/summary_output.html
output: all_extracted_output.txt


**FORMAT EXTRACTed all_extracted_output TXT DATA INTO SUITABLE FORMAT (USE LLM OR WRITE SCRIPT TO AUTOMATE, see doc format in chroma_ingestion.py)**

INPUT EXTRACTED DATA INTO VECTOR DATABASE FOR RAG:

4. run chroma_ingestion.py to input into vector database

5. run python -m streamlit run app.py to start the app

