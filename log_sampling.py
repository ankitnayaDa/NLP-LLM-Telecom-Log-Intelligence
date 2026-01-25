import random
import numpy as np

def get_sample_logs(df, clusters, cluster_id, log_col="clean_log", n=3, seed=42):
    """
    Returns N sample raw log lines belonging to a specific cluster.

    Parameters:
    - df: original dataframe containing raw logs
    - clusters: array of cluster IDs (same order as df)
    - cluster_id: target cluster ID
    - log_col: column name containing raw/clean logs
    - n: number of samples to return
    """

    # Get indices belonging to this cluster
    cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]

    if not cluster_indices:
        return []

    # Sample safely
    random.seed(seed)
    selected_indices = random.sample(
        cluster_indices,
        min(n, len(cluster_indices))
    )
    return df.iloc[selected_indices][log_col].tolist()

def get_top_keywords(kmeans_model, vectorizer, cluster_id, top_n=100):
    """
    Returns the top-N keywords for a given cluster
    based on TF-IDF centroid weights.
    """
    # Get centroid for the cluster
    centroid = kmeans_model.cluster_centers_[cluster_id]
    # Get feature names from TF-IDF
    feature_names = vectorizer.get_feature_names_out()
    # Get indices of top TF-IDF weights
    top_indices = np.argsort(centroid)[-top_n:][::-1]
    # Map indices to feature names
    top_keywords = [feature_names[i] for i in top_indices]
    return top_keywords