# gpt_utils.py

import openai
import streamlit as st
import os

# Use secret if available, otherwise try environment variable
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not openai.api_key:
    st.warning("⚠️ OpenAI API key not found. GPT-based insights will be skipped.")

def generate_section_insight(section, data_snippet):
    if not openai.api_key:
        return "⚠️ Skipped: No API key found."

    try:
        prompt = f"""
        You are an expert energy analyst. Analyze this dashboard section: {section}.
        Data Sample:
        {data_snippet}
        Give 3-5 bullet points of key findings, patterns, anomalies, or recommendations.
        Keep the language simple and insights crisp.
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
