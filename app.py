import streamlit as st
import nltk
import torch
import os
import gdown
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('punkt_tab')

# Function to download the model from Google Drive
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)

# Load Pre-trained Model
def load_model(model_path, vocab_size, embedding_dim=50, hidden_dim=100, output_dim=2):
    model = TextLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Streamlit App
def main():
    st.title("Easy Model Deployment App")

    # Model URL and Path
    model_url = "https://drive.google.com/uc?export=download&id=1Ry9rh_ygoSl_j5wNNNA2NMHIX_0Gw_7Y"
    model_path = "model.pth"

    # Download Model
    st.write("Checking for model...")
    download_model(model_url, model_path)
    st.write("Model downloaded successfully!")

    # Load Model
    vocab_size = 10000
    st.write("Loading model...")
    model = load_model(model_path, vocab_size)
    st.write("Model loaded successfully!")

    # Add an interface for tokenizing text input
    user_input = st.text_input("Enter some text:")
    if user_input:
        st.write("Tokenizing text...")
        tokens = word_tokenize(user_input)
        st.write("Tokens:", tokens)

    # Add a button for model prediction
    if st.button("Predict"):
        # Replace with your prediction logic
        st.write(f"Model Prediction for input: '{user_input}'")

if __name__ == "__main__":
    main()
