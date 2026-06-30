import streamlit as st
import pandas as pd
import joblib
import os
from google import genai

# 1. Setup API
# Replace 'YOUR_API_KEY_HERE' with your actual key
client = genai.Client(api_key='Gemini API KEY')

# 2. Build or Load the ML Brain (Classification)
@st.cache_resource
def get_ml_brain():
    model = joblib.load('mental_health_model.pkl')
    vectorizer = joblib.load('text_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = get_ml_brain()

# 3. Web Page UI
st.set_page_config(page_title="Mental Health Agentic AI", page_icon="🧠")
st.title("🧠 Agentic Mental Health Companion")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Handle Input & Generate Response
if prompt := st.chat_input("I'm here to listen..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Classify Emotion
    prediction = model.predict(vectorizer.transform([prompt]))[0]
    
    # Generate Empathetic Response
    agent_prompt = f"""
    You are a supportive, empathetic mental health companion. 
    The user is expressing feelings related to: {prediction}.
    Provide a compassionate, safe, and helpful response, 
    and suggest one small, healthy coping strategy. 
    Keep it concise.
    User input: {prompt}
    """
    
    with st.spinner("Thinking..."):
        # This uses the new recommended model 'gemini-2.5-flash'
        # If this fails, change to 'gemini-2.0-flash'
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=agent_prompt,
        )

    st.session_state.messages.append({"role": "assistant", "content": response.text})

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
