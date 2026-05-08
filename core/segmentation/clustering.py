import numpy as np
import pandas as pd
import logging
import pickle
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_is_fitted
from .features import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusteringModel:
    """
    Handles user segmentation using K-Means clustering.
    Provides utility to find optimal cluster count using silhouette scores.
    """

    def __init__(self, n_clusters: Optional[int] = None):
        """
        Initialize the model.
        
        Args:
            n_clusters: Number of clusters to use. If None, should be found via find_optimal_k.
        """
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.cluster_sizes_: Dict[int, int] = {}
        
        if n_clusters is not None:
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    def find_optimal_k(self, X: np.ndarray, k_range=range(2, 8)) -> int:
        """
        Fits KMeans for each k in k_range and returns the one with the highest silhouette score.
        """
        best_score = -1
        best_k = 2
        scores = {}

        logger.info(f"Starting search for optimal K in range {list(k_range)}...")

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            scores[k] = score
            
            logger.info(f"K={k} | Silhouette Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k

        self.n_clusters = best_k
        logger.info(f"Optimal K found: {best_k} with score {best_score:.4f}")
        return best_k

    def fit(self, X: np.ndarray, k: Optional[int] = None) -> 'ClusteringModel':
        """
        Trains the KMeans model. If k is None, calls find_optimal_k.
        """
        if k is not None:
            self.n_clusters = k
        elif self.n_clusters is None:
            logger.info("k not provided and self.n_clusters is None. Searching for optimal k...")
            self.find_optimal_k(X)
        
        logger.info(f"Fitting KMeans with n_clusters={self.n_clusters}...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(X)
        
        # Store cluster sizes from training data
        unique, counts = np.unique(self.kmeans.labels_, return_counts=True)
        self.cluster_sizes_ = dict(zip(unique.tolist(), counts.tolist()))
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts cluster assignments for the given data.
        Calls self.kmeans.predict(X). Raises RuntimeError if not fitted.
        """
        if self.kmeans is None:
            raise RuntimeError("ClusteringModel must be fitted before calling predict().")
        
        try:
            check_is_fitted(self.kmeans)
        except Exception:
            raise RuntimeError("ClusteringModel must be fitted before calling predict().")
            
        return self.kmeans.predict(X)

    def cluster_summary(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Returns a table with cluster ID, centroid values per feature, and cluster size.
        Useful for naming and interpreting clusters.
        """
        if self.kmeans is None:
            raise RuntimeError("ClusteringModel must be fitted before generating a summary.")
            
        centroids = self.kmeans.cluster_centers_
        
        # Create summary DataFrame
        df = pd.DataFrame(centroids, columns=feature_names)
        
        # Insert Cluster ID and Size at the beginning
        df.insert(0, 'cluster_id', range(self.n_clusters))
        df.insert(1, 'size', [self.cluster_sizes_.get(i, 0) for i in range(self.n_clusters)])
        
        return df

    def save(self, path: str):
        """
        Serializes the fitted model state.
        """
        if self.kmeans is None:
            raise RuntimeError("Model must be fitted before saving.")
            
        try:
            check_is_fitted(self.kmeans)
        except Exception:
            raise RuntimeError("Model must be fitted before saving.")
        
        state = {
            'kmeans': self.kmeans,
            'cluster_sizes': self.cluster_sizes_
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'ClusteringModel':
        """
        Loads the model state.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        kmeans_obj = state['kmeans']
        instance = cls(n_clusters=kmeans_obj.n_clusters)
        instance.kmeans = kmeans_obj
        instance.cluster_sizes_ = state.get('cluster_sizes', {})
        return instance

    def save_artifacts(self, extractor: FeatureExtractor) -> bytes:
        """
        Pickles both the ClusteringModel state and the FeatureExtractor together 
        into a single bytes object for DB storage.
        """
        if self.kmeans is None:
            raise RuntimeError("ClusteringModel must be fitted before saving artifacts.")
            
        state = {
            'kmeans': self.kmeans,
            'cluster_sizes': self.cluster_sizes_,
            'extractor': extractor
        }
        return pickle.dumps(state)

    @classmethod
    def load_artifacts(cls, artifact_bytes: bytes) -> Tuple['ClusteringModel', FeatureExtractor]:
        """
        Loads both the ClusteringModel and the FeatureExtractor from bytes.
        
        Returns:
            Tuple[ClusteringModel, FeatureExtractor]: The reconstructed instances.
        """
        state = pickle.loads(artifact_bytes)
        
        kmeans_obj = state['kmeans']
        instance = cls(n_clusters=kmeans_obj.n_clusters)
        instance.kmeans = kmeans_obj
        instance.cluster_sizes_ = state.get('cluster_sizes', {})
        
        return instance, state['extractor']
