import streamlit as st
import pandas as pd
import joblib
from google import genai
from google.genai import types

# 1. Setup API (Securely fetching the key)
if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

client = st.session_state.gemini_client

# 2. Load the Deep Learning Brain
@st.cache_resource
def get_ml_brain():
    model = joblib.load('mental_health_dl_model.pkl')
    vectorizer = joblib.load('text_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = get_ml_brain()

# 3. Web Page UI Setup
st.set_page_config(page_title="Mental Health Agentic AI", page_icon="🧠")
st.title("🧠 Agentic Mental Health Companion")

# --- THE MEMORY BACKPACK ---
# If this is a brand new conversation, create the "phone call" and save it.
if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a supportive, empathetic mental health companion. Always suggest one small, healthy coping strategy. Keep it concise."
        )
    )
    # We also need to keep track of the text to draw it on the screen
    st.session_state.ui_messages = []

# 4. Handle Input & Generate Response
if prompt := st.chat_input("I'm here to listen..."):
    # Save and show user message on screen
    st.session_state.ui_messages.append({"role": "user", "content": prompt})
    
    # Classify Emotion using your Neural Network
    prediction = model.predict(vectorizer.transform([prompt]))[0]
    
    # We secretly whisper the emotion to the AI before passing the user's message!
    agent_prompt = f"[System Note: The user's underlying emotion is classified as '{prediction}']\nUser says: {prompt}"
    
    # Send the message into the continuous "phone call" (Memory)
    with st.spinner("Thinking..."):
        response = st.session_state.chat_session.send_message(agent_prompt)

    # Save the AI's response on screen
    st.session_state.ui_messages.append({"role": "assistant", "content": response.text})

# 5. Display the Chat History on the screen
for message in st.session_state.ui_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])