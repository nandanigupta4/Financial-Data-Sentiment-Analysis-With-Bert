
# Financial Data Sentiment Analysis with BERT

This project aims to perform sentiment analysis on financial data using the BERT model. BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art natural language processing model that can capture the context of words from both left and right, and generate high-quality word embeddings. 

## Dataset

The dataset used for this project is the Financial PhraseBank, which contains over 10,000 sentences from financial news articles, labeled with positive, negative, or neutral sentiment. The dataset can be downloaded from [here](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10).

## Model

The model used for this project is the finBERT model, which is a fine-tuned version of BERT for financial domain. The finBERT model is trained on a large corpus of financial news and reports, and can achieve state-of-the-art results on financial sentiment analysis tasks. The finBERT model can be downloaded from [here](https://huggingface.co/ProsusAI/finbert).

## Requirements

The project requires the following libraries and packages:

- transformers
- torch
- pandas
- numpy

## Usage

The project consists of the following files:

- Finanicial Data Classification With Bert.ipynb: The main notebook that contains the code for data preprocessing, model training, and evaluation.
- utils.py: A helper script that contains some utility functions for data loading and processing.
- test.csv: A sample test file that contains some sentences for sentiment analysis.

To run the project, follow these steps:

1. Clone the repository and navigate to the project folder.
2. Install the required libraries and packages using pip or conda.
3. Open the Finanicial Data Classification With Bert.ipynb notebook in Jupyter or Google Colab.
4. Run the cells in the notebook to load the data, train the model, and evaluate the results.
5. To test the model on your own sentences, modify the test.csv file and run the last cell in the notebook.
