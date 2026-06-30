🧠 Agentic AI Mental Health Companion

## 📌 Overview
This project is an advanced, personalized mental health support chatbot designed to provide empathetic, context-aware conversations. Moving beyond basic conversational bots, this system employs an "Agentic AI" architecture. It first uses a Machine Learning classification model to detect the user's emotional state. Then it uses Google's Generative AI (Gemini) to craft supportive, clinically informed responses and coping strategies dynamically.

## 🚀 Tech Stack
* **Language:** Python 3
* **Frontend/UI:** Streamlit
* **Machine Learning:** Scikit-Learn (Logistic Regression, TF-IDF Vectorization)
* **Generative AI:** Google Gemini API (`gemini-2.5-flash`)
* **Data Processing:** Pandas, NumPy

## ⚙️ Architecture & Workflow
1. **Phase 0: Emotion Classification**
   * The user's text input is vectorized using `TfidfVectorizer`.
   * A trained `LogisticRegression` model processes the input to classify the underlying emotional state (e.g., Anxiety, Depression, Stress).
2. **Phase 1: Agentic Context Generation**
   * The detected emotion acts as the "context constraint" for the Generative AI.
3. **Phase 2: Empathetic Generation**
   * The Gemini model takes the user's raw input and the ML-detected emotion to generate a highly personalized response, concluding with a specific, actionable coping strategy.

## 🛠️ How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install streamlit pandas scikit-learn google-genai joblib`
3. Add your Gemini API key to Streamlit secrets.
4. Run the app: `python -m streamlit run app.py`

You can start the app: https://mental-health-chatbot-ai.streamlit.app/

## 👨‍💻 Developer
**Baranidharan T S**
