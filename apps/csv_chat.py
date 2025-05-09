import streamlit as st
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="ESG Data Chat Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .assistant-message {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Set environment variable manually
os.environ['GROQ_API_KEY'] = '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'

# Initialize Groq client
client = Groq()

# Load the CSV data
@st.cache_data
def load_data():
    return pd.read_csv('final.csv')

df = load_data()

def get_llm_response(prompt, df_info):
    system_prompt = f"""You are a helpful assistant analyzing ESG (Environmental, Social, and Governance) data.
    Here's the information about the dataset:
    {df_info}
    
    Please provide insights and answer questions about this data. Be specific and use numbers when available."""
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="gemma2-9b-it",
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar with dataset info
with st.sidebar:
    st.header(" Dataset Information")
    st.write(f" Number of records: {len(df)}")
    st.write(f"Number of features: {len(df.columns)}")
    
    with st.expander(" Available Columns"):
        for col in df.columns:
            st.write(f"- {col}")
    
    with st.expander("Basic Statistics"):
        st.dataframe(df.describe())

# Main content
st.title("🌍 ESG Data Chat Assistant")
st.markdown("""
This chat assistant helps you analyze and understand ESG (Environmental, Social, and Governance) data.
Ask questions about trends, patterns, or specific metrics in the dataset.
""")

# Add chat type selector
chat_type = st.radio(
    "Select Chat Type",
    ["General Analysis", "Data Insights"],
    horizontal=True
)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Customize prompt based on chat type
if chat_type == "General Analysis":
    default_prompt = "Ask about the ESG data (e.g., 'What are the average ESG risk scores?' or 'Show me the distribution of controversy levels')"
else:
    default_prompt = "Ask for specific insights (e.g., 'What are the key trends?' or 'Identify companies with high environmental risks')"

# Chat input
if prompt := st.chat_input(default_prompt):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare dataset information with customized context based on chat type
    if chat_type == "General Analysis":
        df_info = f"""
        Dataset Summary:
        - Shape: {df.shape}
        - Columns: {', '.join(df.columns)}
        - Numeric Statistics:\n{df.describe().to_string()}
        - Sample Data:\n{df.head().to_string()}
        """
    else:
        df_info = f"""
        Dataset Insights Focus:
        - Key Metrics: ESG Risk Scores, Controversy Levels
        - Risk Distribution: {df['ESG Risk Level'].value_counts().to_dict()}
        - Controversy Analysis: {df['Controversy Level'].value_counts().to_dict()}
        - Risk Score Statistics:\n{df[['Total ESG Risk score', 'Environment Risk Score', 'Social Risk Score', 'Governance Risk Score']].describe().to_string()}
        """

    # Get LLM response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            response = get_llm_response(prompt, df_info)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Data viewer section
st.markdown("---")
with st.expander("🔍 View Dataset", expanded=False):
    st.dataframe(df, use_container_width=True)

# Download section
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("###  Download Data")
    st.markdown("Download the complete ESG dataset for offline analysis.")
with col2:
    csv = df.to_csv(index=False)
    st.download_button(
        label=" Download CSV",
        data=csv,
        file_name="esg_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""

""", unsafe_allow_html=True)