# LSTM Text Generation — Generative AI Engineer Practical Test

## 1. Task

This project implements a word-level text generator using:

- Public-domain Shakespeare text
- Text cleaning and tokenization
- Input → next-token sequence preparation
- Keras Embedding layer
- LSTM layer
- Dense softmax output layer
- Adam optimizer
- Sparse categorical crossentropy
- Training/validation split
- Early stopping
- Best-model checkpointing
- Temperature-based text generation

The structure follows the interview task requirements.

## 2. Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- Internet connection for the first dataset download
- TensorFlow

## 3. Setup on Windows

Open Command Prompt in this folder.

Create a virtual environment:

```bat
python -m venv venv
```

Activate it:

```bat
venv\Scripts\activate
```

Upgrade pip:

```bat
python -m pip install --upgrade pip
```

Install dependencies:

```bat
pip install -r requirements.txt
```

Verify TensorFlow:

```bat
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 4. Run the project

```bat
python text_generation.py
```

For a faster first test:

```bat
python text_generation.py --epochs 2 --seq-len 20 --max-vocab 5000 --lstm-units 128
```

For a stronger run:

```bat
python text_generation.py --epochs 15 --seq-len 30 --max-vocab 10000 --lstm-units 256
```

Training time depends on CPU/GPU and machine specifications.

## 5. Dataset

The script automatically downloads:

Shakespeare's Complete Works from Project Gutenberg:

https://www.gutenberg.org/files/100/100-0.txt

The downloaded file is saved as:

`dataset/shakespeare.txt`

## 6. Preprocessing

The script:

1. Converts text to lowercase.
2. Removes punctuation and non-letter characters.
3. Collapses extra spaces.
4. Splits text into word tokens.
5. Builds a vocabulary.
6. Converts words to integer IDs.
7. Creates fixed-length input sequences.
8. Uses the following word as the target.

Example:

```text
Input:
to be or not

Target:
to
```

With a sequence length of 30, each training example contains 30 words and predicts word 31.

## 7. Model architecture

```text
Input sequence
      ↓
Embedding
      ↓
LSTM
      ↓
Dense + Softmax
      ↓
Next-word probability
```

The model uses:

- Embedding dimension: 128
- LSTM units: 256
- Optimizer: Adam
- Loss: SparseCategoricalCrossentropy
- Metric: Accuracy

Sparse categorical crossentropy is used instead of explicitly creating one-hot target vectors because it is more memory-efficient for a large vocabulary.

## 8. Training and validation

The tokenized dataset is split sequentially:

- 90% training
- 10% validation

Early stopping monitors validation loss. The best model is saved to:

`models/best_lstm.keras`

The final model is saved to:

`models/final_lstm.keras`

## 9. Text generation

The model starts with a seed such as:

```text
to be or not to be
```

It predicts one word, appends that word to the context, predicts another word, and repeats.

Temperature controls randomness:

- `0.5` → more conservative/repetitive
- `0.8` → balanced
- `1.0` → more random

## 10. Output

After training:

`outputs/generated_samples.txt`

contains generated text for multiple seed inputs.

## 11. Bonus experiment

Run different sequence lengths or LSTM sizes.

Example:

```bat
python text_generation.py --epochs 10 --seq-len 20 --lstm-units 128
python text_generation.py --epochs 10 --seq-len 40 --lstm-units 256
```

Record:

- validation loss
- validation accuracy
- training time
- quality/coherence of generated text

### Suggested observation

A longer sequence can provide more context but increases computational cost. A larger LSTM can learn more complex patterns but generally requires more memory and training time.

## 12. Interview explanation

If asked "Explain your project", say:

> I developed a word-level LSTM text generation system using TensorFlow and Keras. I first cleaned and tokenized a public-domain Shakespeare dataset, converted words into integer IDs, and created fixed-length input sequences where the target was the next word. I used an Embedding layer followed by an LSTM and a Dense softmax layer for next-token prediction. I used Adam optimization and sparse categorical crossentropy, split the data into training and validation sets, and used early stopping and model checkpointing. Finally, I generated text iteratively from different seed phrases using temperature-based sampling.

## 13. Deliverables

Before submission, make sure these exist:

- `text_generation.py`
- `requirements.txt`
- `README.md`
- `dataset/shakespeare.txt`
- `models/best_lstm.keras`
- `models/final_lstm.keras`
- `outputs/generated_samples.txt`
