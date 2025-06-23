# gpt_utils.py

import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_section_insight(section, data_snippet):
    try:
        prompt = f"""
        You are an expert energy analyst. Analyze this dashboard section: {section}.
        Data Sample:
        {data_snippet}
        Give 3-5 bullet points of key findings, patterns, anomalies, or recommendations.Also keep the description short and crisp in easy language.
        """
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert energy dashboard assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT failed: {e}"

def generate_full_report(insight_list):
    return "\n\n".join([f"## {title}\n{content}" for title, content in insight_list])
