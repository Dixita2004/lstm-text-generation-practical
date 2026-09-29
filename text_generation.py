
"""
LSTM Text Generation - Generative AI Engineer Practical Test

This project follows the interview task:
1. Load a public-domain text dataset.
2. Lowercase and remove punctuation.
3. Tokenize into words.
4. Create input -> next-token training pairs.
5. Train an Embedding + LSTM + Dense(softmax) model.
6. Use early stopping and checkpoints.
7. Generate text from multiple seed inputs.

Run:
    python text_generation.py

Optional:
    python text_generation.py --epochs 10 --seq-len 30 --max-vocab 10000
"""

import argparse
import re
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models


DATA_URL = "https://www.gutenberg.org/files/100/100-0.txt"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = DATA_DIR / "shakespeare.txt"


def download_dataset():
    """Download Shakespeare's Complete Works if it is not already present."""
    DATA_DIR.mkdir(exist_ok=True)
    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 100_000:
        print(f"[OK] Dataset already exists: {DATA_FILE}")
        return

    print("[INFO] Downloading public-domain Shakespeare dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print(f"[OK] Downloaded: {DATA_FILE}")


def clean_text(text):
    """
    Lowercase and remove punctuation/non-letter characters.
    Multiple spaces are collapsed into one.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_sequences(tokens, seq_len):
    """
    For tokens [w1, w2, ..., wn], create:
      input  = [wi ... wi+seq_len-1]
      target = w(i+seq_len)

    The target is the next token after the input sequence.
    """
    inputs = []
    targets = []

    for i in range(len(tokens) - seq_len):
        inputs.append(tokens[i:i + seq_len])
        targets.append(tokens[i + seq_len])

    return inputs, targets


def prepare_data(text, seq_len, max_vocab):
    """
    Build a word-level vocabulary and integer sequences.

    Returns:
        x: shape (samples, seq_len)
        y: shape (samples,)
        word_to_id
        id_to_word
    """
    text = clean_text(text)
    tokens = text.split()

    # Keep the most frequent words.
    counts = {}
    for word in tokens:
        counts[word] = counts.get(word, 0) + 1

    sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    vocabulary = ["<UNK>"] + [w for w, _ in sorted_words[:max_vocab - 1]]

    word_to_id = {word: i for i, word in enumerate(vocabulary)}
    id_to_word = {i: word for word, i in word_to_id.items()}

    ids = [word_to_id.get(word, 0) for word in tokens]

    # Split sequentially: first 90% training, last 10% validation.
    split = int(len(ids) * 0.90)
    train_ids = ids[:split]
    val_ids = ids[split:]

    train_inputs, train_targets = build_sequences(train_ids, seq_len)
    val_inputs, val_targets = build_sequences(val_ids, seq_len)

    x_train = np.asarray(train_inputs, dtype=np.int32)
    y_train = np.asarray(train_targets, dtype=np.int32)
    x_val = np.asarray(val_inputs, dtype=np.int32)
    y_val = np.asarray(val_targets, dtype=np.int32)

    print(f"[INFO] Total tokens: {len(tokens):,}")
    print(f"[INFO] Vocabulary size: {len(vocabulary):,}")
    print(f"[INFO] Training samples: {len(x_train):,}")
    print(f"[INFO] Validation samples: {len(x_val):,}")

    return x_train, y_train, x_val, y_val, word_to_id, id_to_word


def build_model(vocab_size, seq_len, embedding_dim=128, lstm_units=256):
    """
    Model architecture:
        Input -> Embedding -> LSTM -> Dense(softmax)
    """
    model = models.Sequential([
        layers.Input(shape=(seq_len,)),
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="embedding"
        ),
        layers.LSTM(lstm_units, name="lstm"),
        layers.Dense(vocab_size, activation="softmax", name="next_word")
    ])

    # Sparse categorical crossentropy avoids constructing huge one-hot targets.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()
    return model


def sample_with_temperature(probabilities, temperature=0.8):
    """Sample an index from a probability distribution."""
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities + 1e-8) / temperature
    probabilities = np.exp(probabilities - np.max(probabilities))
    probabilities /= np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)


def generate_text(model, seed, word_to_id, id_to_word, seq_len,
                  num_words=40, temperature=0.8):
    """Generate new words iteratively from a seed."""
    seed_clean = clean_text(seed)
    seed_words = seed_clean.split()

    if not seed_words:
        raise ValueError("Seed text must contain at least one word.")

    generated = seed_words[:]

    for _ in range(num_words):
        context = generated[-seq_len:]
        encoded = [word_to_id.get(word, 0) for word in context]

        if len(encoded) < seq_len:
            encoded = [0] * (seq_len - len(encoded)) + encoded

        x = np.asarray([encoded], dtype=np.int32)
        probabilities = model.predict(x, verbose=0)[0]
        next_id = sample_with_temperature(probabilities, temperature)
        next_word = id_to_word[next_id]

        # Avoid printing the unknown token.
        if next_word != "<UNK>":
            generated.append(next_word)

    return " ".join(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--max-vocab", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lstm-units", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    download_dataset()

    raw_text = DATA_FILE.read_text(encoding="utf-8", errors="ignore")

    x_train, y_train, x_val, y_val, word_to_id, id_to_word = prepare_data(
        raw_text,
        seq_len=args.seq_len,
        max_vocab=args.max_vocab,
    )

    model = build_model(
        vocab_size=len(word_to_id),
        seq_len=args.seq_len,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
    )

    checkpoint_path = MODEL_DIR / "best_lstm.keras"

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n[INFO] Training started...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=1,
    )

    # Save final model.
    final_model_path = MODEL_DIR / "final_lstm.keras"
    model.save(final_model_path)

    # Save vocabulary.
    vocab_path = MODEL_DIR / "vocabulary.json"
    import json
    vocab_path.write_text(
        json.dumps(word_to_id, indent=2),
        encoding="utf-8"
    )

    seeds = [
        "to be or not to be",
        "the king shall",
        "love is",
    ]

    output_lines = []
    output_lines.append("LSTM TEXT GENERATION - SAMPLE OUTPUTS\n")

    for seed in seeds:
        generated = generate_text(
            model,
            seed,
            word_to_id,
            id_to_word,
            seq_len=args.seq_len,
            num_words=40,
            temperature=args.temperature,
        )
        output_lines.append(f"Seed: {seed}")
        output_lines.append(f"Generated: {generated}")
        output_lines.append("")

    output_file = OUTPUT_DIR / "generated_samples.txt"
    output_file.write_text("\n".join(output_lines), encoding="utf-8")

    print("\n[OK] Training complete.")
    print(f"[OK] Best model:  {checkpoint_path}")
    print(f"[OK] Final model: {final_model_path}")
    print(f"[OK] Samples:     {output_file}")


if __name__ == "__main__":
    main()
