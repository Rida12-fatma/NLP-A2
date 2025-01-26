# Language Model Training and Text Generation Application

This application demonstrates how to build, train, and utilize a language model using PyTorch, NLTK, and Streamlit. It uses the classic text dataset "Alice's Adventures in Wonderland" from Project Gutenberg as the training corpus. The application includes data preprocessing, model training, text generation, and an interactive Streamlit interface for testing the language model.

## Features
- Downloads and preprocesses text data.
- Builds and trains a neural network-based language model.
- Generates text based on a given prompt.
- Interactive user interface using Streamlit for real-time text generation.

## Installation
### Prerequisites
Ensure you have Python 3.8 or above installed. Install the required Python libraries by running:

```bash
pip install -r requirements.txt
```

### Required Libraries
- requests
- nltk
- numpy
- torch
- streamlit

## Usage

1. **Download the Dataset**
   The dataset is automatically downloaded from Project Gutenberg and saved locally as `alice_dataset.txt`.

2. **Run the Application**
   To start the application, run the following command in your terminal:

   ```bash
   streamlit run app.py
   ```

   This will launch the Streamlit app in your web browser.

3. **Interact with the Application**
   Use the interface to input a starting text prompt. The model will generate a sequence of words based on the input.

## Project Structure
- `app.py`: The main application script.
- `alice_dataset.txt`: The processed text dataset (automatically generated).
- `model.pth`: The trained PyTorch model (saved during training).
- `requirements.txt`: A list of Python dependencies.

## How it Works
### Data Preprocessing
- Tokenizes the dataset and removes punctuation and stopwords.
- Converts words into numerical indices using a vocabulary dictionary.
- Creates input sequences for training.

### Model Architecture
The language model is implemented using PyTorch and consists of:
- **Embedding Layer**: Converts word indices into dense vector representations.
- **LSTM Layers**: Captures sequential patterns in text.
- **Fully Connected Layer**: Predicts the next word in the sequence.

### Training
- The model is trained using cross-entropy loss and the Adam optimizer.
- The dataset is split into sequences, and the model learns to predict the next word in a sequence.

### Text Generation
- Uses the trained model to generate text by predicting one word at a time based on the input prompt.
- Incorporates randomness using a softmax probability distribution.

## Example Output
### Input Prompt:
```text
harry potter is
```

### Generated Text:
```text
harry potter is instantly directions came frogfootman oh get hands said hatter feel know soldiers shouted king queen reading finish appeared...
```

## Customization
### Hyperparameters
- `embedding_dim`: Dimension of word embeddings.
- `hidden_dim`: Number of hidden units in LSTM.
- `sequence_length`: Length of input sequences for training.
- `batch_size`: Number of sequences per batch.
- `epochs`: Number of training iterations.

### Dataset
Replace the dataset URL in `app.py` to use a different text corpus.

## Future Enhancements
- Add support for different languages.
- Use pre-trained embeddings like GloVe or FastText.
- Implement beam search for more coherent text generation.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it as per the license terms.

