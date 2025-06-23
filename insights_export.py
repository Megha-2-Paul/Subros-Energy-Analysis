# insights_export.py

import base64
import streamlit as st
from datetime import datetime

def download_report_button(insight_list):
    full_report = "\n\n".join([f"## {title}\n{content}" for title, content in insight_list])
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    b64 = base64.b64encode(full_report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="Energy_Report_{now}.txt">📄 Download Full Report</a>'
    st.markdown(href, unsafe_allow_html=True)
