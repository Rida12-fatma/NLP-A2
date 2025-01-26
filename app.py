import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
import requests

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset (Alice's Adventures in Wonderland)
@st.cache_data
def load_dataset():
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    response = requests.get(url)
    return response.text

data = load_dataset()

# Preprocess the dataset
def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    # Remove punctuation and special characters
    tokens = [re.sub(r'\W+', '', token) for token in tokens if re.sub(r'\W+', '', token)]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Add special token for unknown words
    tokens.append('<UNK>')
    return tokens

tokens = preprocess_text(data)

# Create vocabulary and mappings
vocab = list(set(tokens))
word2index = {word: i for i, word in enumerate(vocab)}
index2word = {i: word for i, word in enumerate(vocab)}

# Create sequences
sequence_length = 5
sequences = [
    tokens[i:i + sequence_length] for i in range(len(tokens) - sequence_length)
]
input_sequences = np.array([
    [word2index[word] for word in sequence] for sequence in sequences
])

# Define the Language Model
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, prev_state):
        x = self.embedding(x)
        x, state = self.lstm(x, prev_state)
        x = self.fc(x)
        return x, state

    def init_state(self, batch_size=1):
        return (torch.zeros(2, batch_size, self.lstm.hidden_size),
                torch.zeros(2, batch_size, self.lstm.hidden_size))

# Hyperparameters and Model Initialization
embedding_dim = 50
hidden_dim = 100
vocab_size = len(vocab)
model = LanguageModel(vocab_size, embedding_dim, hidden_dim)

# Load the trained model
try:
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
except FileNotFoundError:
    st.warning("Trained model file not found. Please ensure 'model.pth' is in the same directory.")

# Generate text using the model
def generate_text(model, start_text, max_length=50):
    words = start_text.split()
    state_h, state_c = model.init_state(batch_size=1)

    for _ in range(max_length):
        x = torch.tensor([[word2index.get(w, word2index['<UNK>']) for w in words]], dtype=torch.long)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(index2word[word_index])

    return ' '.join(words)

# Streamlit Interface
st.title("Language Model Text Generator")
st.write("Generate text based on Alice's Adventures in Wonderland!")

start_text = st.text_input("Enter a starting phrase:", value="Alice was")
max_length = st.slider("Select the maximum length of generated text:", 10, 100, 50)

if st.button("Generate Text"):
    if start_text:
        generated_text = generate_text(model, start_text, max_length)
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter a starting phrase.")
