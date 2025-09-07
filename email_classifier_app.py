import streamlit as st
import joblib

# Load the saved pipeline (vectorizer + model)
with open("spam_pipeline.pkl", "rb") as f:
    model = joblib.load(f)

st.title("📩 Email Classifier")

# -------------------------
# Create a text area and predict button
# -------------------------
email_text = st.text_area(
    "Paste Your email here",  # label
    height=200
)

if st.button("Predict Spam"):
    try:
        # Predict class
        prediction = model.predict([email_text])[0]

        # Predict probabilities
        proba = model.predict_proba([email_text])[0]
        spam_index = list(model.classes_).index("spam")
        spam_prob = proba[spam_index]
        not_spam_prob = 1 - spam_prob

        # Display results
        result = "✅ Not Spam" if prediction == "ham" else "❌ Spam"
        st.subheader(result)
        st.write(f"Not Spam Probability: {not_spam_prob:.2%}")
        st.write(f"Spam Probability: {spam_prob:.2%}")

    except Exception as e:
        st.error(f"Error: {e}")
