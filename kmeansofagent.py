import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time

def load_credit_card_data(filepath='creditcard.csv'):
    """Loads the real-world credit card dataset."""
    df = pd.read_csv(filepath)

    # Drop the target column ('Class') and any non-numeric columns
    df = df.drop(columns=['Class', 'Time'], errors='ignore')

    # Optional: Normalize/standardize if needed (skipped here as it's already PCA'd)
    return df

def find_best_kmeans_clusters(data, k_range=range(2, 11), random_state=42):
    """Find the best K using silhouette score and return clustering results."""
    best_k = None
    best_score = -1
    best_model = None
    silhouette_scores = {}

    for k in k_range:
        print(f"Clustering with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labels = kmeans.fit_predict(data)

        score = silhouette_score(data, labels, sample_size=10000)  # sample for speed
        silhouette_scores[k] = score

        if score > best_score:
            best_k = k
            best_score = score
            best_model = kmeans

    return best_k, best_score, best_model, silhouette_scores

def cluster_real_data():
    print("Loading dataset...")
    df = load_credit_card_data()

    print(f"Dataset loaded with shape: {df.shape}")
    print("Finding best number of clusters...")

    start_time = time.time()
    best_k, best_score, model, scores = find_best_kmeans_clusters(df)
    elapsed = time.time() - start_time

    df['cluster'] = model.predict(df)

    print(f"\n Best number of clusters: {best_k}")
    print(f"Silhouette Score: {best_score:.4f}")
    print(f"Time taken: {elapsed:.2f} seconds")

    return df, best_k, best_score, scores

def plot_silhouette_scores(scores):
    plt.figure(figsize=(8, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.title("Silhouette Score by Number of Clusters (k)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    clustered_df, best_k, best_score, silhouette_scores = cluster_real_data()
    plot_silhouette_scores(silhouette_scores)

    print("\nClustered Data Sample:")
    print(clustered_df.head())
    # Optional: clustered_df.to_csv("clustered_creditcard.csv", index=False)
