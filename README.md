LLM Telecom Log Intelligence
=============================

This project implements an end-to-end pipeline for telecom log intelligence:

1. **Log Preprocessing & Vectorization**
   - Clean and tokenize raw MME logs
   - Vectorize using custom TF-IDF and transformer embeddings

2. **Semantic Clustering**
   - Group similar signaling behaviors using HDBSCAN
   - Extract top keywords to characterize each group

3. **Representative Log Sampling**
   - Extract sample logs per cluster for contextual relevance

4. **LLM-Based Explanation Layer**
   - Use a Large Language Model to generate human-readable
     explanations for each cluster
   - Explanations include procedure descriptions, behavior
     classification, and troubleshooting guidance

Pipeline Overview
-------------
- Preprocess logs and vectorize text using NLP techniques.
- Cluster log entries to find semantic groups.
- Extract top keywords and representative samples per cluster.
- Format prompts and call an LLM (e.g., Google Gemini) to generate cluster explanations.
- Output cluster explanations in JSON.

Deliverables
-------------
- Code modules for each step.
- Explanations written by an LLM for each meaningful cluster.
- Documentation explaining how to run and interpret results.

Requirements
-------------
- Python 3.8+
- Packages: pandas, scikit-learn, hdbscan, sentence-transformers, transformers,
	spacy, python-dotenv, google-genai

Install
-------
Install packages (example):

```bash
pip install pandas scikit-learn hdbscan sentence-transformers transformers spacy python-dotenv google-genai
python -m spacy download en_core_web_sm
```

Configuration
-------------
- Create a `.env` file in the project root with your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```
- The prompt template used for cluster explanations is at `prompts/cluster_explainer_prompt.txt`.

Data
-----
- Default sample log file: `data/mme_1.002.log` (the pipeline currently uses the first 50 non-empty lines).

How it works (high level)
-------------------------
- `lib/lib.py` — preprocessing helpers and a fast spaCy-based pipeline.
- `nlp_log_vectorizer.py` — reads the log, cleans it, runs a zero-shot classifier,
	generates embeddings with `sentence-transformers`, and triggers clustering.
- `log_clustering.py` — performs HDBSCAN clustering on the embeddings and adds
	cluster IDs and confidence scores to the dataframe.
- `log_sampling.py` — utilities to sample logs from clusters and extract
	top keywords (TF-IDF helpers included).
- `llm_log_explainer.py` — builds prompts per cluster (using the sample logs and
	extracted keywords) and calls Google Gemini (`google.genai`) to generate
	natural-language explanations. Outputs are written to `output/cluster_explanations.json`.
- `pipeline.py` — simple entrypoint which runs the explainer.

Usage
------
- Run the full pipeline (after setting `.env`):

```bash
python pipeline.py
```

- Or run pieces interactively in a REPL / notebook by importing functions from
	the modules above. `nlp_log_vectorizer.nlp_log_embedding_vectorizer()` returns
	a pandas DataFrame with `embeddings`, `cluster`, and `cluster_confidence`.

Notes & customization
----------------------
- `llm_log_explainer.py` uses `MODEL_NAME = "gemini-2.5-flash-lite"` by
	default — change if you need a different model.
- `generate_cluster_prompts()` limits sample logs per cluster via `max_logs`.
- The explainer includes basic retry logic for rate limits and server errors.
- The preprocessing pipeline currently keeps only `NOUN` and `PROPN` lemmas —
	adjust `lib/fast_nlp_pipeline()` if you want different token selections.

Example LLM Output
------
- Cluster explanations: `output/cluster_explanations.json` (JSON mapping ofcluster ID -> explanation text).
```json
{
  "UE Attach (Normal)": [
    "Represents standard attach flow.",
    "Behavior is normal.",
    "No specific action required."
  ],
  "Inactivity / Abnormal Release": [
    "Indicates abnormal session termination due to inactivity.",
    "Possible root causes: inactivity timer expiry, radio drop.",
    "Recommend verifying ECM timers and radio conditions."
  ]
}
```json

Contributing
------------
- Suggestions, bug reports, and improvements welcome. Small, focused changes
	that preserve the repo layout are easiest to review.

License
-------
- No license specified.