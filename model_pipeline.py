import os
import re
import math
from collections import Counter

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# SETTINGS — SAME AS NOTEBOOK
# ============================================================

MAX_VOCAB_SIZE = 30000
MAX_LEN = 256

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
ATTENTION_DIM = 64

NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TOKENIZER
# ============================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text):
    text = str(text).lower()
    return re.findall(
        r"\b\w+\b|[^\w\s]",
        text
    )


# ============================================================
# REBUILD THE SAME VOCABULARY
# ============================================================

print("Loading IMDB dataset to rebuild vocabulary...")

dataset = load_dataset("stanfordnlp/imdb")

train_data = dataset["train"]

# Same 80/20 split used in notebook.
# We only need the training portion to build vocabulary.
from sklearn.model_selection import train_test_split

texts = train_data["text"]
labels = train_data["label"]

train_texts, _, _, _ = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

counter = Counter()

for text in train_texts:
    counter.update(tokenize(text))


vocab = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1
}

for idx, (token, frequency) in enumerate(
    counter.most_common(MAX_VOCAB_SIZE - 2),
    start=2
):
    vocab[token] = idx


print("Vocabulary size:", len(vocab))


# ============================================================
# TEXT ENCODING
# ============================================================

def encode_text(text, vocab):
    tokens = tokenize(text)

    return [
        vocab.get(token, vocab[UNK_TOKEN])
        for token in tokens
    ]


def encode_and_pad(text, vocab, max_len=MAX_LEN):

    encoded = encode_text(text, vocab)

    if len(encoded) > max_len:
        encoded = encoded[:max_len]

    if len(encoded) < max_len:
        encoded += [
            vocab[PAD_TOKEN]
        ] * (max_len - len(encoded))

    return encoded


# ============================================================
# SIMPLE RNN
# ============================================================

class SimpleRNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes=2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, input_ids):

        embedded = self.embedding(input_ids)

        output, hidden = self.rnn(
            embedded
        )

        last_output = output[:, -1, :]

        logits = self.fc(
            last_output
        )

        return logits


# ============================================================
# LSTM
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.0,
        num_classes=2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, input_ids):

        embedded = self.embedding(
            input_ids
        )

        output, (hidden, cell) = self.lstm(
            embedded
        )

        last_hidden = hidden[-1]

        logits = self.fc(
            last_hidden
        )

        return logits


# ============================================================
# GRU
# ============================================================

class GRUClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.0,
        num_classes=2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, input_ids):

        embedded = self.embedding(
            input_ids
        )

        output, hidden = self.gru(
            embedded
        )

        last_hidden = hidden[-1]

        logits = self.fc(
            last_hidden
        )

        return logits


# ============================================================
# BiLSTM
# ============================================================

class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.0,
        num_classes=2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            num_classes
        )

    def forward(self, input_ids):

        embedded = self.embedding(
            input_ids
        )

        output, (hidden, cell) = self.lstm(
            embedded
        )

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined_hidden = torch.cat(
            (
                forward_hidden,
                backward_hidden
            ),
            dim=1
        )

        logits = self.fc(
            combined_hidden
        )

        return logits


# ============================================================
# ADDITIVE ATTENTION
# ============================================================

class AdditiveAttention(nn.Module):

    def __init__(
        self,
        hidden_dim,
        attention_dim
    ):
        super().__init__()

        self.W = nn.Linear(
            hidden_dim,
            attention_dim
        )

        self.v = nn.Linear(
            attention_dim,
            1,
            bias=False
        )

    def forward(
        self,
        lstm_outputs,
        mask=None
    ):

        energy = torch.tanh(
            self.W(lstm_outputs)
        )

        scores = self.v(
            energy
        ).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                -1e9
            )

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_outputs
        ).squeeze(1)

        return context, attention_weights


# ============================================================
# ATTENTION LSTM
# ============================================================

class AttentionLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        attention_dim,
        num_classes=2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.attention = AdditiveAttention(
            hidden_dim=hidden_dim,
            attention_dim=attention_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, input_ids):

        mask = (
            input_ids != 0
        )

        embedded = self.embedding(
            input_ids
        )

        lstm_outputs, (hidden, cell) = self.lstm(
            embedded
        )

        context, attention_weights = (
            self.attention(
                lstm_outputs,
                mask
            )
        )

        logits = self.fc(
            context
        )

        return logits, attention_weights


# ============================================================
# TRANSFORMER — FROM SCRATCH
# ============================================================

class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q,
        K,
        V,
        mask=None
    ):

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        d_k = Q.size(-1)

        scores = (
            scores /
            math.sqrt(d_k)
        )

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                -1e9
            )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        context = torch.matmul(
            attention_weights,
            V
        )

        return (
            context,
            attention_weights
        )


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = (
            d_model // num_heads
        )

        self.q_linear = nn.Linear(
            d_model,
            d_model
        )

        self.k_linear = nn.Linear(
            d_model,
            d_model
        )

        self.v_linear = nn.Linear(
            d_model,
            d_model
        )

        self.out_linear = nn.Linear(
            d_model,
            d_model
        )

        self.attention = (
            ScaledDotProductAttention()
        )

    def split_heads(self, x):

        batch_size, seq_len, _ = (
            x.size()
        )

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def combine_heads(self, x):

        batch_size, num_heads, seq_len, head_dim = (
            x.size()
        )

        x = x.transpose(1, 2)

        x = x.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

        return x

    def forward(
        self,
        x,
        mask=None
    ):

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        context, attention_weights = (
            self.attention(
                Q,
                K,
                V,
                mask
            )
        )

        context = self.combine_heads(
            context
        )

        output = self.out_linear(
            context
        )

        return (
            output,
            attention_weights
        )


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=256
    ):
        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        seq_len = x.size(1)

        return (
            x +
            self.pe[:, :seq_len, :]
        )


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                d_model,
                d_ff
            ),
            nn.ReLU(),
            nn.Linear(
                d_ff,
                d_model
            )
        )

    def forward(self, x):
        return self.network(x)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):
        super().__init__()

        self.self_attention = (
            MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads
            )
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        mask=None
    ):

        attention_output, attention_weights = (
            self.self_attention(
                x,
                mask=mask
            )
        )

        x = self.norm1(
            x +
            self.dropout(
                attention_output
            )
        )

        ff_output = self.feed_forward(x)

        x = self.norm2(
            x +
            self.dropout(
                ff_output
            )
        )

        return (
            x,
            attention_weights
        )


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_len=256,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0
        )

        self.positional_encoding = (
            PositionalEncoding(
                d_model=d_model,
                max_len=max_len
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(
            d_model
        )

    def forward(
        self,
        input_ids,
        mask=None
    ):

        x = self.embedding(
            input_ids
        )

        x = (
            x *
            math.sqrt(
                self.embedding.embedding_dim
            )
        )

        x = self.positional_encoding(x)

        x = self.dropout(x)

        all_attention_weights = []

        for layer in self.layers:

            x, attention_weights = (
                layer(
                    x,
                    mask=mask
                )
            )

            all_attention_weights.append(
                attention_weights
            )

        x = self.norm(x)

        return (
            x,
            all_attention_weights
        )


def create_padding_mask(input_ids):

    mask = (
        input_ids != 0
    )

    return (
        mask
        .unsqueeze(1)
        .unsqueeze(2)
    )


class TransformerClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_len=256,
        dropout=0.1,
        num_classes=2
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout
        )

        self.classifier = nn.Linear(
            d_model,
            num_classes
        )

    def masked_mean_pooling(
        self,
        x,
        input_ids
    ):

        mask = (
            input_ids != 0
        ).float()

        mask = mask.unsqueeze(-1)

        x = x * mask

        summed = x.sum(dim=1)

        counts = mask.sum(dim=1)

        counts = counts.clamp(
            min=1.0
        )

        pooled = (
            summed / counts
        )

        return pooled

    def forward(self, input_ids):

        padding_mask = (
            create_padding_mask(
                input_ids
            )
        )

        encoder_output, attention_weights = (
            self.encoder(
                input_ids,
                mask=padding_mask
            )
        )

        pooled = self.masked_mean_pooling(
            encoder_output,
            input_ids
        )

        logits = self.classifier(
            pooled
        )

        return (
            logits,
            attention_weights
        )


# ============================================================
# LOAD ALL 7 TRAINED MODELS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def load_checkpoint(model, filename):

    path = os.path.join(
        BASE_DIR,
        filename
    )

    state = torch.load(
        path,
        map_location=DEVICE
    )

    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    return model


print("Loading trained models...")


rnn_model = load_checkpoint(
    SimpleRNN(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ),
    "rnn_final.pt"
)


lstm_model = load_checkpoint(
    LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        dropout=0.0,
        num_classes=NUM_CLASSES
    ),
    "lstm_final.pt"
)


gru_model = load_checkpoint(
    GRUClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        dropout=0.0,
        num_classes=NUM_CLASSES
    ),
    "gru_final.pt"
)


bilstm_model = load_checkpoint(
    BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        dropout=0.0,
        num_classes=NUM_CLASSES
    ),
    "bilstm_final.pt"
)


attention_lstm_model = load_checkpoint(
    AttentionLSTM(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        attention_dim=ATTENTION_DIM,
        num_classes=NUM_CLASSES
    ),
    "attention_lstm_final.pt"
)


transformer_model = load_checkpoint(
    TransformerClassifier(
        vocab_size=len(vocab),
        d_model=128,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_len=MAX_LEN,
        dropout=0.1,
        num_classes=NUM_CLASSES
    ),
    "transformer_final.pt"
)


# ============================================================
# DISTILBERT
# ============================================================

DISTILBERT_NAME = (
    "distilbert-base-uncased"
)

distilbert_tokenizer = (
    AutoTokenizer.from_pretrained(
        DISTILBERT_NAME
    )
)

distilbert_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        DISTILBERT_NAME,
        num_labels=2
    )
)

distilbert_model = (
    distilbert_model.to(DEVICE)
)

distilbert_path = os.path.join(
    BASE_DIR,
    "distilbert_best_final.pt"
)

distilbert_model.load_state_dict(
    torch.load(
        distilbert_path,
        map_location=DEVICE
    )
)

distilbert_model.eval()


print("All 7 models loaded successfully.")


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_all_models(text):

    results = {}

    # --------------------------------------------------------
    # Custom models
    # --------------------------------------------------------

    tokens = tokenize(text)

    token_ids = encode_text(
        text,
        vocab
    )

    tokens = tokens[:MAX_LEN]
    token_ids = token_ids[:MAX_LEN]

    padded_ids = (
        token_ids +
        [vocab[PAD_TOKEN]] *
        (MAX_LEN - len(token_ids))
    )

    input_tensor = torch.tensor(
        padded_ids,
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)


    standard_models = {
        "RNN": rnn_model,
        "LSTM": lstm_model,
        "GRU": gru_model,
        "BiLSTM": bilstm_model
    }


    for model_name, model in (
        standard_models.items()
    ):

        with torch.no_grad():

            logits = model(
                input_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            prediction = torch.argmax(
                logits,
                dim=1
            ).item()

        results[model_name] = {
            "prediction": prediction,
            "probabilities":
                probabilities[0]
                .cpu()
                .numpy()
                .tolist()
        }


    # --------------------------------------------------------
    # Attention LSTM
    # --------------------------------------------------------

    with torch.no_grad():

        logits, attention_weights = (
            attention_lstm_model(
                input_tensor
            )
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            logits,
            dim=1
        ).item()

    attention = (
        attention_weights[
            0,
            :len(tokens)
        ]
        .cpu()
        .numpy()
        .tolist()
    )

    results["Attention LSTM"] = {
        "prediction": prediction,
        "probabilities":
            probabilities[0]
            .cpu()
            .numpy()
            .tolist(),
        "tokens": tokens,
        "attention_weights": attention
    }


    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    with torch.no_grad():

        logits, attention_matrices = (
            transformer_model(
                input_tensor
            )
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            logits,
            dim=1
        ).item()

    transformer_attention = []

    for layer_attention in (
        attention_matrices
    ):

        matrix = (
            layer_attention[
                0,
                :,
                :len(tokens),
                :len(tokens)
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        transformer_attention.append(
            matrix
        )

    results["Transformer"] = {
        "prediction": prediction,
        "probabilities":
            probabilities[0]
            .cpu()
            .numpy()
            .tolist(),
        "tokens": tokens,
        "attention":
            transformer_attention
    }


    # --------------------------------------------------------
    # DistilBERT
    # --------------------------------------------------------

    encoding = (
        distilbert_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
    )

    input_ids = (
        encoding["input_ids"]
        .to(DEVICE)
    )

    attention_mask = (
        encoding["attention_mask"]
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = distilbert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            logits,
            dim=1
        ).item()

    results["DistilBERT"] = {
        "prediction": prediction,
        "probabilities":
            probabilities[0]
            .cpu()
            .numpy()
            .tolist()
    }


    return results