import numpy as np
import hdbscan
  
# Embedding-based Clustering
def log_embedding_clustering(log):
    print(f"\n=== HDBSCAN Clustering ===\n")
    X = np.vstack(log["embeddings"].values)
    # Cluster
    cluster = hdbscan.HDBSCAN(min_cluster_size=5,min_samples=3,metric='euclidean')
    log["cluster"] = cluster.fit_predict(X)
    log["cluster_confidence"] = cluster.probabilities_
    print(log.head(5))
    return log


    