```markdown
# Text Generation with Language Model

This repository contains a text generation application built using a language model implemented in PyTorch and deployed with Streamlit.

## Dataset

The dataset used is the SMS Spam Collection dataset.

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Usage

1. Run the `app.py` file:

```sh
streamlit run app.py
```

2. Open your web browser and go to the URL provided by Streamlit.

3. Enter a text prompt in the input box and click on the "Generate Text" button to see the model's generated text.

## Model

The language model is implemented using PyTorch and consists of an embedding layer, LSTM layers, and a fully connected layer.

## Training

The model is trained on the SMS Spam Collection dataset. The training loop and hyperparameters are defined in the `app.py` file.

## License

This project is licensed under the MIT License.
```
