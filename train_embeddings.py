# -*- coding: utf-8 -*-
"""
CENG 442 - Assignment 1
Section 8: Train Word2Vec & FastText on combined cleaned_text
"""

from gensim.models import Word2Vec, FastText
import pandas as pd
from pathlib import Path

# =========================================================
# 1. Files: cleaned 2-column Excel outputs
# =========================================================
files = [
    "labeled-sentiment_2col.xlsx",
    "test__1__2col.xlsx",
    "train__3__2col.xlsx",
    "train-00000-of-00001_2col.xlsx",
    "merged_dataset_CSV__1__2col.xlsx",
]

# =========================================================
# 2. Build sentence list for training
# =========================================================
sentences = []
for f in files:
    print(f"📖 Reading {f}")
    df = pd.read_excel(f, usecols=["cleaned_text"])
    # Split each row into tokens (list of words)
    sentences.extend(df["cleaned_text"].astype(str).str.split().tolist())

print(f"✅ Total sentences (rows) loaded: {len(sentences)}")

# =========================================================
# 3. Train Word2Vec
# =========================================================
Path("embeddings").mkdir(exist_ok=True)

print("\n🔹 Training Word2Vec model...")
w2v = Word2Vec(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=3,
    sg=1,             # skip-gram
    negative=10,
    epochs=10
)
w2v.save("embeddings/word2vec.model")
print("✅ Word2Vec model saved to embeddings/word2vec.model")

# =========================================================
# 4. Train FastText
# =========================================================
print("\n🔹 Training FastText model...")
ft = FastText(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=3,
    sg=1,
    min_n=3,  # subword min length
    max_n=6,  # subword max length
    epochs=10
)
ft.save("embeddings/fasttext.model")
print("✅ FastText model saved to embeddings/fasttext.model")

print("\n🎉 All embeddings trained and saved successfully!")
