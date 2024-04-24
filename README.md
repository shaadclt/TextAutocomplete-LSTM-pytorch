# Text Autocompletion using LSTM in PyTorch
This repository contains a Jupyter Notebook demonstrating text autocompletion using Long Short-Term Memory (LSTM) networks implemented in PyTorch.

## Introduction
Text autocompletion is a useful feature in various natural language processing applications. LSTM networks, a type of recurrent neural network (RNN), are well-suited for this task due to their ability to capture long-range dependencies in sequential data.

## Features
- Train an LSTM model to predict the next word in a sequence based on input text.
- Evaluate model performance using training and validation loss.
- Generate predictions for user-provided input sentences.

## Getting Started
To run the notebook, follow these steps:

1. Clone this repository to your local machine.
```bash
git clone https://github.com/shaadclt/TextAutocomplete-LSTM-pytorch.git
```
2. Open the Jupyter Notebook **autocomplete_LSTM_pytorch.ipynb** in your preferred environment.
3.Execute each cell in the notebook sequentially.

## Dataset
The dataset used for training consists of text data extracted from a Wikipedia article. The text is preprocessed to remove special characters and non-English words.

## Model Architecture
The LSTM model architecture consists of an embedding layer, followed by multiple LSTM layers and a fully connected layer. Dropout is applied to prevent overfitting.

## Training
The model is trained using the training dataset with a specified sequence length, batch size, learning rate, and number of epochs. Training progress is monitored using average loss.

## Evaluation
Model performance is evaluated using both training and validation loss. Lower loss values indicate better predictive performance.

## Usage
After training the model, users can input partial sentences, and the model will predict the next word based on the provided context.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
1. This project is inspired by the need for intelligent text processing solutions.
2. Special thanks to the PyTorch community for their invaluable contributions.
