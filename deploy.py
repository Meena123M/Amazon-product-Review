import streamlit as st
import pickle
import numpy as np

# Load the saved model and vectorizer
with open('multinomial_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit UI
st.title("🧠 Amazon Review Sentiment Classifier")
st.write("Enter a product review, and the model will predict if it's **Positive** or **Negative**.")

# Input box
user_input = st.text_area("Enter your review here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Preprocess input
        transformed_input = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(transformed_input)

        # Show result
        sentiment = "✅ Positive" if prediction[0] == 1 else "❌ Negative"
        st.subheader("Prediction:")
        st.success(sentiment)
