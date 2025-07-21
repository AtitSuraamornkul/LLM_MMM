import streamlit as st
import json
import re
import pandas as pd

def extract_vega_dataset_from_html(html_path, chart_id):
    """
    Extracts the first dataset from the Vega-Lite spec for a given chart_id in an HTML file.
    Returns a pandas DataFrame.
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        # Find the Vega-Lite spec for the given chart_id
        pattern = re.compile(
            r'chart-embed id="' + re.escape(chart_id) + r'".*?JSON\.parse\("({.*?})"\);',
            re.DOTALL
        )
        
        match = pattern.search(html)
        if not match:
            st.warning(f"Could not find Vega-Lite spec for chart id '{chart_id}'")
            return None
            
        vega_json_str = match.group(1).encode('utf-8').decode('unicode_escape')
        vega_spec = json.loads(vega_json_str)
        
        # Get the dataset (first item in the 'datasets' dict)
        if 'datasets' not in vega_spec:
            st.warning(f"No datasets found in Vega spec for chart id '{chart_id}'")
            return None
            
        data = list(vega_spec['datasets'].values())[0]
        return pd.DataFrame(data)
    
    except Exception as e:
        st.error(f"Error extracting data for chart '{chart_id}': {str(e)}")
        return None