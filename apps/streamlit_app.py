
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import requests

# Set page config
st.set_page_config(
    page_title="ESG Risk Level Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar selection
options = ['Prediction', 'Model Metrics Chat']
option = st.sidebar.selectbox('Select an option', options)

# Load model, scaler, and metrics data
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('esg_scaler.pkl')
    return model, scaler

@st.cache_data
def load_metrics():
    return pd.read_csv('model_metrics_comparison.csv', index_col=0)

model, scaler = load_model_and_scaler()
metrics_df = load_metrics()

# --- PREDICTION TAB ---
if option == 'Prediction':
    st.title("ESG Risk Level Predictor")
    st.markdown("Enter the ESG metrics to predict the risk level.")

    with st.form("prediction_form"):
        st.subheader("Enter ESG Metrics")
        total_esg = st.number_input("Total ESG Risk Score (0-100)", 0.0, 100.0, 50.0)
        env_risk = st.number_input("Environment Risk Score (0-100)", 0.0, 100.0, 50.0)
        gov_risk = st.number_input("Governance Risk Score (0-100)", 0.0, 100.0, 50.0)
        social_risk = st.number_input("Social Risk Score (0-100)", 0.0, 100.0, 50.0)
        controversy_level = st.number_input("Controversy Level (0-5)", 0.0, 5.0, 2.0)
        controversy_score = st.number_input("Controversy Score (0-100)", 0.0, 100.0, 50.0)
        risk_percentile = st.number_input("ESG Risk Percentile (0-100)", 0.0, 100.0, 50.0)

        submit_button = st.form_submit_button("Predict Risk Level")

    if submit_button:
        features = np.array([
            total_esg, env_risk, gov_risk, social_risk,
            controversy_level, controversy_score, risk_percentile
        ]).reshape(1, -1)

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        probability = float(max(prediction_proba[0]))

        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ESG Risk Level", int(prediction[0]))
        with col2:
            st.metric("Confidence", f"{probability:.2%}")

        risk_levels = {
            0: "Very Low Risk", 1: "Low Risk", 2: "Medium Risk",
            3: "High Risk", 4: "Very High Risk"
        }
        st.info(f"Interpretation: {risk_levels[int(prediction[0])]}")

# --- MODEL METRICS CHAT TAB ---
elif option == 'Model Metrics Chat':
    st.title("Model Metrics Chat Assistant")
    st.markdown("Chat with the model metrics data to get insights about model performance.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask about model metrics (e.g., 'Which model has the highest accuracy?'):")

    if user_input:
        response = ""
        if "highest" in user_input.lower() and "accuracy" in user_input.lower():
            best_model = metrics_df['Accuracy'].idxmax()
            accuracy = metrics_df.loc[best_model, 'Accuracy']
            response = f"The model with the highest accuracy is **{best_model}** with an accuracy of **{accuracy:.4f}**."
        elif "compare" in user_input.lower():
            models = [m.strip() for m in user_input.lower().split("compare")[1].split("and")]
            models_cleaned = [m for m in metrics_df.index if m.lower() in models]
            if len(models_cleaned) == 2:
                comparison = metrics_df.loc[models_cleaned]
                response = f"Comparison:\n\n{comparison.to_markdown()}"
            else:
                response = "Please specify two valid models to compare."
        elif "lowest" in user_input.lower() or "worst" in user_input.lower():
            worst_model = metrics_df['Accuracy'].idxmin()
            accuracy = metrics_df.loc[worst_model, 'Accuracy']
            response = f"The model with the lowest accuracy is **{worst_model}** with an accuracy of **{accuracy:.4f}**."
        else:
            response = (
                "I can help you with:\n- Best/worst model by accuracy\n"
                "- Comparing two models\n- Checking precision, recall, or F1 scores"
            )

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Assistant", response))

    for role, message in st.session_state.chat_history:
        st.markdown(f"**{role}:** {message}")

    st.subheader("Model Metrics Table")
    st.dataframe(metrics_df.style.highlight_max(axis=0))

    # ESG definitions
    st.markdown("""
    ---
    ### About ESG Risk Levels
    - **Very Low Risk (0)**: Minimal ESG risk exposure  
    - **Low Risk (1)**: Limited ESG risk exposure  
    - **Medium Risk (2)**: Moderate ESG risk exposure  
    - **High Risk (3)**: Significant ESG risk exposure  
    - **Very High Risk (4)**: Severe ESG risk exposure  
    """)

    # --- Groq Assistant Section ---
    os.environ['GROQ_API_KEY'] = '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'

    @st.cache_data
    def load_final_csv():
        return pd.read_csv('final.csv')

    df = load_final_csv()

    st.title("ESG Data Chat Assistant")
    st.markdown("Ask questions about ESG trends, patterns, or specific metrics in the dataset.")

    chat_type = st.radio("Select Chat Type", ["General Analysis", "Data Insights"], horizontal=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if chat_type == "General Analysis":
        default_prompt = "Ask about average ESG scores or distribution of controversy levels."
    else:
        default_prompt = "Ask for key trends or companies with high environmental risk."

    if prompt := st.chat_input(default_prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if chat_type == "General Analysis":
            df_info = f"""
            Dataset Summary:
            - Shape: {df.shape}
            - Columns: {', '.join(df.columns)}
            - Numeric Stats:\n{df.describe().to_string()}
            """
        else:
            df_info = f"""
            ESG Focused Summary:
            - Risk Distribution: {df['ESG Risk Level'].value_counts().to_dict()}
            - Controversy: {df['Controversy Level'].value_counts().to_dict()}
            - Scores Stats:\n{df[['Total ESG Risk score', 'Environment Risk Score', 'Social Risk Score', 'Governance Risk Score']].describe().to_string()}
            """

        def get_llm_response(prompt, df_info):
            headers = {
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "gemma2-9b-it",
                "messages": [
                    {"role": "system", "content": f"You are an assistant analyzing ESG data:\n{df_info}"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1024
            }

            try:
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Error: {str(e)}"

        assistant_response = get_llm_response(prompt, df_info)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
=======
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import requests

# Set page config
st.set_page_config(
    page_title="ESG Risk Level Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar selection
options = ['Prediction', 'Model Metrics Chat']
option = st.sidebar.selectbox('Select an option', options)

# Load model, scaler, and metrics data
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(r'C:\Users\Shreyansh Singh\Desktop\splunk_hackathon\encodings\xgb_model.pkl')
    scaler = joblib.load(r'C:\Users\Shreyansh Singh\Desktop\splunk_hackathon\encodings\esg_scaler.pkl')
    return model, scaler

@st.cache_data
def load_metrics():
    return pd.read_csv(r'C:\Users\Shreyansh Singh\Desktop\splunk_hackathon\data\model_metrics_comparison.csv', index_col=0)

model, scaler = load_model_and_scaler()
metrics_df = load_metrics()

# --- PREDICTION TAB ---
if option == 'Prediction':
    st.title("ESG Risk Level Predictor")
    st.markdown("Enter the ESG metrics to predict the risk level.")

    with st.form("prediction_form"):
        st.subheader("Enter ESG Metrics")
        total_esg = st.number_input("Total ESG Risk Score (0-100)", 0.0, 100.0, 50.0)
        env_risk = st.number_input("Environment Risk Score (0-100)", 0.0, 100.0, 50.0)
        gov_risk = st.number_input("Governance Risk Score (0-100)", 0.0, 100.0, 50.0)
        social_risk = st.number_input("Social Risk Score (0-100)", 0.0, 100.0, 50.0)
        controversy_level = st.number_input("Controversy Level (0-5)", 0.0, 5.0, 2.0)
        controversy_score = st.number_input("Controversy Score (0-100)", 0.0, 100.0, 50.0)
        risk_percentile = st.number_input("ESG Risk Percentile (0-100)", 0.0, 100.0, 50.0)

        submit_button = st.form_submit_button("Predict Risk Level")

    if submit_button:
        features = np.array([
            total_esg, env_risk, gov_risk, social_risk,
            controversy_level, controversy_score, risk_percentile
        ]).reshape(1, -1)

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        probability = float(max(prediction_proba[0]))

        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ESG Risk Level", int(prediction[0]))
        with col2:
            st.metric("Confidence", f"{probability:.2%}")

        risk_levels = {
            0: "Very Low Risk", 1: "Low Risk", 2: "Medium Risk",
            3: "High Risk", 4: "Very High Risk"
        }
        st.info(f"Interpretation: {risk_levels[int(prediction[0])]}")

# --- MODEL METRICS CHAT TAB ---
elif option == 'Model Metrics Chat':
    st.title("Model Metrics Chat Assistant")
    st.markdown("Chat with the model metrics data to get insights about model performance.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask about model metrics (e.g., 'Which model has the highest accuracy?'):")

    if user_input:
        response = ""
        if "highest" in user_input.lower() and "accuracy" in user_input.lower():
            best_model = metrics_df['Accuracy'].idxmax()
            accuracy = metrics_df.loc[best_model, 'Accuracy']
            response = f"The model with the highest accuracy is **{best_model}** with an accuracy of **{accuracy:.4f}**."
        elif "compare" in user_input.lower():
            models = [m.strip() for m in user_input.lower().split("compare")[1].split("and")]
            models_cleaned = [m for m in metrics_df.index if m.lower() in models]
            if len(models_cleaned) == 2:
                comparison = metrics_df.loc[models_cleaned]
                response = f"Comparison:\n\n{comparison.to_markdown()}"
            else:
                response = "Please specify two valid models to compare."
        elif "lowest" in user_input.lower() or "worst" in user_input.lower():
            worst_model = metrics_df['Accuracy'].idxmin()
            accuracy = metrics_df.loc[worst_model, 'Accuracy']
            response = f"The model with the lowest accuracy is **{worst_model}** with an accuracy of **{accuracy:.4f}**."
        else:
            response = (
                "I can help you with:\n- Best/worst model by accuracy\n"
                "- Comparing two models\n- Checking precision, recall, or F1 scores"
            )

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Assistant", response))

    for role, message in st.session_state.chat_history:
        st.markdown(f"**{role}:** {message}")

    st.subheader("Model Metrics Table")
    st.dataframe(metrics_df.style.highlight_max(axis=0))

    # ESG definitions
    st.markdown("""
    ---
    ### About ESG Risk Levels
    - **Very Low Risk (0)**: Minimal ESG risk exposure  
    - **Low Risk (1)**: Limited ESG risk exposure  
    - **Medium Risk (2)**: Moderate ESG risk exposure  
    - **High Risk (3)**: Significant ESG risk exposure  
    - **Very High Risk (4)**: Severe ESG risk exposure  
    """)

    # --- Groq Assistant Section ---
    os.environ['GROQ_API_KEY'] = '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'

    @st.cache_data
    def load_final_csv():
        return pd.read_csv(r'C:\Users\Shreyansh Singh\Desktop\splunk_hackathon\data\final.csv')

    df = load_final_csv()

    st.title("ESG Data Chat Assistant")
    st.markdown("Ask questions about ESG trends, patterns, or specific metrics in the dataset.")

    chat_type = st.radio("Select Chat Type", ["General Analysis", "Data Insights"], horizontal=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if chat_type == "General Analysis":
        default_prompt = "Ask about average ESG scores or distribution of controversy levels."
    else:
        default_prompt = "Ask for key trends or companies with high environmental risk."

    if prompt := st.chat_input(default_prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if chat_type == "General Analysis":
            df_info = f"""
            Dataset Summary:
            - Shape: {df.shape}
            - Columns: {', '.join(df.columns)}
            - Numeric Stats:\n{df.describe().to_string()}
            """
        else:
            df_info = f"""
            ESG Focused Summary:
            - Risk Distribution: {df['ESG Risk Level'].value_counts().to_dict()}
            - Controversy: {df['Controversy Level'].value_counts().to_dict()}
            - Scores Stats:\n{df[['Total ESG Risk score', 'Environment Risk Score', 'Social Risk Score', 'Governance Risk Score']].describe().to_string()}
            """

        def get_llm_response(prompt, df_info):
            headers = {
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "gemma2-9b-it",
                "messages": [
                    {"role": "system", "content": f"You are an assistant analyzing ESG data:\n{df_info}"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1024
            }

            try:
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Error: {str(e)}"

        assistant_response = get_llm_response(prompt, df_info)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
>>>>>>> aa63b7cda38efaa55c9a033e14c1d57d6c1c089f
