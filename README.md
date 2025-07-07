1. run the mmm analysis (hitachi_mmm.ipynb) -> 

input: MMM dataset
output: output/summary_output.html   |   output/optimization_output.html


2. run the optimization extractor (optim_extract.py) -> 

input: output/optimization_output.html
output: llm_input/llm_input.txt

3. run the summary extractor (summary_extract.py) ->

input: output/summary_output.html
output: summary_output/summary_extract_output.txt


**INPUT EXTRACTED DATA INTO VECTOR DATABASE FOR RAG**

4. **FORMAT EXTRACTED DATA INTO SUITABLE FORMAT FOR VECTOR DATABASE(doc_process.py, doc_process2.py is not automatic)

5. adjust and run ingestion.py to input into vector database

6. run python -m streamlit run app_RAG.py to start the app

