# -*- coding: utf-8 -*-
"""
CENG 442 – Assignment 1
Section 9: Compare Word2Vec and FastText Embeddings
"""

from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# 1. Load models
# =====================================================
print("📂 Loading embeddings...")
w2v = Word2Vec.load("embeddings/word2vec.model")
ft  = FastText.load("embeddings/fasttext.model")
print("✅ Models loaded successfully\n")

# =====================================================
# 2. Vocabulary coverage (per dataset)
# =====================================================
datasets = [
    "labeled-sentiment_2col.xlsx",
    "test__1__2col.xlsx",
    "train__3__2col.xlsx",
    "train-00000-of-00001_2col.xlsx",
    "merged_dataset_CSV__1__2col.xlsx",
]

def lexical_coverage(model, texts):
    vocab = model.wv.key_to_index
    words = [w for t in texts for w in t.split()]
    known = sum(1 for w in words if w in vocab)
    return known / len(words)

rows = []
for f in datasets:
    df = pd.read_excel(f, usecols=["cleaned_text"])
    texts = df["cleaned_text"].astype(str).tolist()
    w2v_cov = lexical_coverage(w2v, texts)
    ft_cov  = lexical_coverage(ft, texts)
    rows.append((Path(f).stem, round(w2v_cov,3), round(ft_cov,3)))
cov_df = pd.DataFrame(rows, columns=["Dataset","Word2Vec","FastText"])
print("== Lexical coverage ==")
print(cov_df.to_string(index=False), "\n")

# =====================================================
# (Optional) Coverage by Domain (bonus)
# =====================================================
import re

def detect_domain(text):
    s = text.lower()
    if re.search(r"\b(apa|trend|azertac|reuters|bloomberg|dha|aa)\b", s):
        return "domnews"
    if re.search(r"\b(rt)\b|@|#|(?:😂|😍|😊|👍|👎|😡|🙂)", s):
        return "domsocial"
    if re.search(r"\b(azn|manat|qiymət|aldım|ulduz|çox yaxşı|çox pis)\b", s):
        return "domreviews"
    return "domgeneral"

domains = {"domnews": [], "domsocial": [], "domreviews": [], "domgeneral": []}

for f in datasets:
    df = pd.read_excel(f, usecols=["cleaned_text"])
    for t in df["cleaned_text"].astype(str).tolist():
        dom = detect_domain(t)
        domains[dom].append(t)

print("== Coverage by Domain ==")
for dom, texts in domains.items():
    if not texts:
        continue
    w2v_cov = lexical_coverage(w2v, texts)
    ft_cov  = lexical_coverage(ft, texts)
    print(f"{dom:12s}  W2V={w2v_cov:.3f}   FT={ft_cov:.3f}")
print()



# =====================================================
# 3. Synonym / antonym similarities
# =====================================================
pairs_syn = [("yaxşı","əla"),("bahalı","qiymətli"),("pis","zəif")]
pairs_ant = [("yaxşı","pis"),("bahalı","ucuz"),("müsbət","mənfi")]

def pair_sim(model, pairs):
    sims = []
    for a,b in pairs:
        if a in model.wv and b in model.wv:
            sims.append(model.wv.similarity(a,b))
    return np.mean(sims) if sims else None

print("== Similarities ==")
print(f"Synonyms  →  Word2Vec={pair_sim(w2v,pairs_syn):.3f},  FastText={pair_sim(ft,pairs_syn):.3f}")
print(f"Antonyms  →  Word2Vec={pair_sim(w2v,pairs_ant):.3f},  FastText={pair_sim(ft,pairs_ant):.3f}\n")

# =====================================================
# 4. Nearest neighbors for inspection
# =====================================================
words = ["yaxşı","pis","bahalı","film","mahnı"]
print("== Nearest Neighbors ==")
for w in words:
    if w in w2v.wv:
        print(f"\n🔹 Word2Vec({w}): {[x for x,_ in w2v.wv.most_similar(w, topn=5)]}")
    if w in ft.wv:
        print(f"🔸 FastText({w}): {[x for x,_ in ft.wv.most_similar(w, topn=5)]}")

# (Optional) domain drift if you train domain-specific models separately: 
# drift(word, model_a, model_b) = 1 - cos(vec_a, vec_b)