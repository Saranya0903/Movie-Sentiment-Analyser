# MovieMind: Movie Review Sentiment Analyser

An NLP project that classifies IMDB movie reviews as positive or negative using seven models, from a Simple RNN to a fine-tuned DistilBERT. A FastAPI service runs all models, and a Gradio interface shows each model's prediction, confidence and attention.

**Author:** Saranya Das (Reg No 25122019, 4 MSCDS), Christ University, Pune Lavasa Campus

## Models and results

All models were trained on the same split of the IMDB dataset (`stanfordnlp/imdb`) and evaluated once on the official 25,000-review test set. Seed: 42.

| Model | Description | Test accuracy |
|---|---|---|
| Simple RNN | Recurrent baseline | 53.044% |
| LSTM | Gated recurrent | 56.036% |
| GRU | Gated recurrent | 79.904% |
| BiLSTM | Bidirectional LSTM | 79.556% |
| Attention LSTM | LSTM with additive attention | 85.032% |
| Transformer | Encoder written from scratch in PyTorch | 82.252% |
| DistilBERT | Pre-trained, fine-tuned | 90.708% |

## Project structure

```
api.py                    FastAPI service (GET /, POST /predict)
model_pipeline.py         Tokeniser, vocabulary, model classes, checkpoint loading, inference
app.py                    Gradio interface (calls the API)
requirements.txt          Python dependencies
rnn_final.pt              Trained weights
lstm_final.pt
gru_final.pt
bilstm_final.pt
attention_lstm_final.pt
transformer_final.pt
distilbert_final.pt
distilbert_best_final.pt  Best-validation DistilBERT checkpoint (used by the app)
```

## Setup

1. Clone the repository. The `.pt` weight files are stored with Git LFS, so install it first:

```
   git lfs install
   git clone https://github.com/Saranya0903/Movie-Sentiment-Analyser.git
   cd Movie-Sentiment-Analyser
   git lfs pull
```

2. Install the dependencies:

```
   pip install -r requirements.txt
```

## Run

Start the API first. The interface needs it to be running.

```
uvicorn api:app --reload
```

The API runs at `http://127.0.0.1:8000`, and interactive docs are at `http://127.0.0.1:8000/docs`.

In a second terminal, start the interface:

```
python app.py
```

Open the local address Gradio prints (usually `http://127.0.0.1:7860`), enter a movie review and click **Analyze My Review**.

**First start:** the pipeline downloads the IMDB dataset to rebuild the vocabulary, and it downloads the `distilbert-base-uncased` tokenizer. This needs an internet connection and takes a few minutes. Later starts use the local cache.

## API

`POST /predict`

```json
{ "review": "The acting was brilliant and the story was beautiful." }
```

The response contains each model's predicted label (`0` = negative, `1` = positive) and class probabilities. The Attention LSTM entry also includes the tokens and their attention weights, and the Transformer entry includes attention matrices for every layer and head. An empty review returns an error message.

## How the text is processed

- **Custom models:** lowercase the text, split it into words and punctuation with a regular expression, map tokens to a 30,000-entry vocabulary built from the training split (index 0 is padding, index 1 is unknown), then truncate or pad to 256 tokens.
- **DistilBERT:** its own WordPiece tokeniser, with inputs padded or truncated to 256 tokens.

## Limitations

- The plain RNN and LSTM scored close to chance level. The from-scratch Transformer and DistilBERT finished slightly below their accuracy targets.
- The tokeniser does not remove HTML tags such as `<br />` and splits contractions such as "didn't".
- Reviews longer than 256 tokens are cut from the end.
- The task has only two classes, so neutral and mixed opinions are forced into positive or negative.
- Sarcasm remains difficult for all models.
- Attention weights are an inspection aid, not a full explanation of a prediction.
