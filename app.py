import streamlit as st
import joblib

# Load vectorizer and model
vectorizer = joblib.load('tdidf_vectorizer.pkl')
model = joblib.load('logistic_regression.pkl')

# Streamlit UI
st.title("🧠 Fake News Detection App")
st.markdown("Enter a message below and the model will predict whether it is fake or real.")

# Text input
user_input = st.text_area("✍️ Enter your message here:")

# Predict on button click
if st.button("Predict Real Or Fake"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize and predict
        text_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(text_tfidf)[0]
        if prediction==1:
            st.success(f"**Predicted Sentiment:** Real")
        else:
            st.success(f"**Predicted Sentiment:** Fake")