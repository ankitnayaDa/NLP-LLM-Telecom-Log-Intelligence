import pandas as pd
from transformers import pipeline
from lib.lib import pre_clean,fast_nlp_pipeline
from log_clustering import log_embedding_clustering
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformers import logging
from sentence_transformers import SentenceTransformer

logging.set_verbosity_error()

def nlp_log_embedding_vectorizer():
    with open("data/mme_1.002.log", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    log = pd.DataFrame({"text": lines})
    log = log[log["text"].str.strip() != ""].head(50)

    log["text"] = log["text"].apply(pre_clean)
    log["clean_log"] = fast_nlp_pipeline(log["text"])

    # Zero-shot classification
    labels = [
        "Authentication Failure",
        "Attach Failure",
        "Detach Procedure",
        "TAU Failure",
        "MME Internal Error",
        "Normal Operation"
    ]

    #classifiers
    print(f"\n=== zero-shot-classification ===\n")
    classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli",device=-1)
    results = classifier(log.clean_log.tolist(), labels)
    log["Category"] = [r["labels"][0] for r in results] # type: ignore
    print(log.head(5))

    # Embeddings
    print(f"\n=== SentenceTransformer Embeddings ===\n")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    log["embeddings"] = embedder.encode(log.clean_log.tolist()).tolist()
    print(log.head(5))
    log =log_embedding_clustering(log)
    return log