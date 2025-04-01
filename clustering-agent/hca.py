"""
Hierarchical clustering hyperparameter tuning and evaluation on Iris dataset.

This module finds optimal hyperparameters for hierarchical clustering, evaluates performance
using silhouette score and adjusted Rand index, and visualizes dendrogram and clusters.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from itertools import product

LINKAGE_METHODS = ['single', 'complete', 'average', 'ward']
DISTANCE_METRICS = ['euclidean', 'cityblock', 'cosine']
N_CLUSTERS_OPTIONS = [2, 3, 4]

def evaluate_clustering(labels, X, y_true):
    """
    Compute silhouette and adjusted Rand scores.

    Args:
        labels (array-like): Cluster labels.
        X (array-like): Feature data.
        y_true (array-like): True labels.

    Returns:
        tuple: (silhouette_score, adjusted_rand_index)
    """
    silhouette = silhouette_score(X, labels)
    rand_index = adjusted_rand_score(y_true, labels)
    return silhouette, rand_index

def main():
    """Main function to execute clustering workflow."""
    data = load_iris()
    X, y_true = data.data, data.target

    results = []
    for linkage_method, metric, n_clusters in product(
        LINKAGE_METHODS, DISTANCE_METRICS, N_CLUSTERS_OPTIONS
    ):
        if linkage_method == 'ward' and metric != 'euclidean':
            continue  # Ward linkage only supports euclidean distance explicitly

        Z = linkage(X, method=linkage_method, metric=metric)
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')

        silhouette, rand_index = evaluate_clustering(labels, X, y_true)

        results.append({
            'linkage': linkage_method,
            'metric': metric,
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'rand_index': rand_index
        })

    results_df = pd.DataFrame(results)
    best_params = results_df.sort_values('silhouette', ascending=False).iloc[0]

    print("Best Parameters:")
    print(best_params)

    Z_best = linkage(X, method=best_params.linkage, metric=best_params.metric)
    labels_best = fcluster(Z_best, t=int(best_params.n_clusters), criterion='maxclust')

    plt.figure(figsize=(12, 6))
    dendrogram(
        Z_best,
        truncate_mode='lastp',
        p=20,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Cluster size')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(
        x=X[:, 0], y=X[:, 1], hue=labels_best, palette='Set1', ax=axes[0]
    )
    axes[0].set_title('Cluster Assignments')
    axes[0].set_xlabel(data.feature_names[0])
    axes[0].set_ylabel(data.feature_names[1])

    sns.scatterplot(
        x=X[:, 0], y=X[:, 1], hue=y_true, palette='Set2', ax=axes[1]
    )
    axes[1].set_title('True Labels')
    axes[1].set_xlabel(data.feature_names[0])
    axes[1].set_ylabel(data.feature_names[1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()