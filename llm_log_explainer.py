import json
from dotenv import load_dotenv
from pathlib import Path
import os
import time
from google import genai
from google.genai import errors
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_log_vectorizer import nlp_log_embedding_vectorizer

PROMPT_PATH = Path("prompts/cluster_explainer_prompt.txt")

def load_prompt():
    return PROMPT_PATH.read_text()

def explain_cluster(output_file="output/cluster_explanations.json"):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    MODEL_NAME = "gemini-2.5-flash-lite"

    log = nlp_log_embedding_vectorizer()
    cluster_prompts = generate_cluster_prompts(log)

    all_explanations = {}
    print("Generating content (handling potential rate limits)...")

    for cluster_id, prompt in cluster_prompts.items():
        cluster_key = str(cluster_id)
        explanation_text = None
        success = False

        for attempt in range(3):
            try:
                response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
                explanation_text = response.text or ""
                success = True
                break

            except errors.ClientError as e:
                if "429" in str(e):
                    wait = 5 * (attempt + 1)
                    print(f"Rate limit hit for cluster {cluster_key}. Sleeping {wait}s...")
                    time.sleep(wait)
                else:
                    raise

            except errors.ServerError as e:
                if "503" in str(e) or "overloaded" in str(e).lower():
                    wait = 20 * (attempt + 1)
                    print(
                        f"MODEL OVERLOADED (503) for cluster {cluster_key}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

        if not success:
            explanation_text = "Error: Failed to generate explanation."

        # STORE RESULT PER CLUSTER
        all_explanations[cluster_key] = explanation_text
        print(f"\n=== Cluster {cluster_key} Explanation ===\n")
        print(explanation_text)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_explanations, f, indent=4, ensure_ascii=False)

    print(f"Saved cluster explanations to: {output_file}")
    return all_explanations

def generate_cluster_prompts(df, max_logs=10):
    print(f"\n=== generate_cluster_prompts ===\n")
    cluster_prompts = {}
    for cluster_id in sorted(df.cluster.unique()):
        if cluster_id == -1:
            continue  # skip noise
        print(f"\n=== generate_cluster_prompts ===\n",cluster_id)
        cluster_df = df[df.cluster == cluster_id]
        print(f"\n=== generate_cluster_prompts ===\n",cluster_df)
        sample_logs = (cluster_df["clean_log"].head(max_logs).tolist())
        print(f"\n=== generate_cluster_prompts ===\n",sample_logs)
        keywords = extract_cluster_keywords(df, cluster_id)
        print(f"\n=== generate_cluster_prompts ===\n",keywords)
        cluster_label = f"Cluster-{cluster_id}"
        print(f"\n=== generate_cluster_prompts ===\n",cluster_label)
        prompt = build_cluster_prompt(cluster_label=cluster_label,keywords=keywords,sample_logs=sample_logs)
        cluster_prompts[cluster_id] = prompt
        print(f"\n=== generate_cluster_prompts ===\n",cluster_prompts[cluster_id])
    return cluster_prompts

def build_cluster_prompt(cluster_label,keywords,sample_logs):
    print(f"\n=== build_cluster_prompt ===\n")
    prompt_template = load_prompt()
    prompt = prompt_template.format(cluster_label=cluster_label,keywords=", ".join(keywords),sample_logs="\n".join(sample_logs))
    return prompt

def extract_cluster_keywords(df, cluster_id, top_n=5):
    print(f"\n=== extract_cluster_keywords ===\n")
    cluster_logs = df[df.cluster == cluster_id]["clean_log"].tolist()
    if len(cluster_logs) < 2:
        return []

    vectorizer = TfidfVectorizer(max_features=50,stop_words="english",ngram_range=(1, 2))
    X = vectorizer.fit_transform(cluster_logs)
    scores = X.mean(axis=0).A1  # type: ignore
    terms = vectorizer.get_feature_names_out()

    keywords = sorted(zip(terms, scores),key=lambda x: x[1],reverse=True)[:top_n]
    print(f"\n=== extract_cluster_keywords ===\n",keywords)
    return [k[0] for k in keywords]