import streamlit as st
import joblib

# Load the saved pipeline (vectorizer + model)
with open("spam_pipeline.pkl", "rb") as f:
    model = joblib.load(f)

# --- Define the callback function ---
def clear_text():
    st.session_state.email_text = ""

st.title("📩 Email Classifier")

# -------------------------
# Initialize state
# -------------------------
if "email_text" not in st.session_state:
    st.session_state["email_text"] = ""

# -------------------------
# Text area bound to session_state
# -------------------------
email_text = st.text_area(
    "Paste Your email here",
    height=200,
    key="email_text"  # bind text area to state
)

# -------------------------
# Two buttons side by side
# -------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Predict Spam"):
        try:
            prediction = model.predict([st.session_state["email_text"]])[0]
            proba = model.predict_proba([st.session_state["email_text"]])[0]

            spam_index = list(model.classes_).index("spam")
            spam_prob = proba[spam_index]
            not_spam_prob = 1 - spam_prob

            result = "✅ Not Spam" if prediction == "ham" else "❌ Spam"
            st.subheader(result)
            st.write(f"Not Spam Probability: {not_spam_prob:.2%}")
            st.write(f"Spam Probability: {spam_prob:.2%}")

        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    # --- Conditionally render the clear button ---
    if st.session_state.email_text:
        st.button("Clear", on_click=clear_text)
