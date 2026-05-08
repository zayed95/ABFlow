import pytest
import numpy as np
import os
from sklearn.datasets import make_blobs
from core.segmentation.clustering import ClusteringModel

@pytest.fixture
def blob_data():
    """Generates synthetic data with 3 clusters."""
    X, _ = make_blobs(n_samples=100, centers=3, n_features=5, random_state=42)
    return X

class TestClusteringModel:
    
    def test_find_optimal_k_bounds(self, blob_data):
        """Test that find_optimal_k returns a value within the provided range."""
        model = ClusteringModel()
        k_range = range(2, 6)
        optimal_k = model.find_optimal_k(blob_data, k_range=k_range)
        
        assert optimal_k in k_range
        assert model.n_clusters == optimal_k

    def test_predict_consistency(self, blob_data):
        """Test that predict() on training data matches the labels assigned during fit()."""
        model = ClusteringModel(n_clusters=3)
        model.fit(blob_data)
        
        # Predictions on the same training data should match the internal labels_
        predictions = model.predict(blob_data)
        assert np.array_equal(predictions, model.kmeans.labels_)

    def test_save_load_persistence(self, blob_data, tmp_path):
        """Test save/load round-trip for model parameters and internal state."""
        model = ClusteringModel(n_clusters=3)
        model.fit(blob_data)
        original_predictions = model.predict(blob_data)
        original_sizes = model.cluster_sizes_
        
        # Use tmp_path for a clean test environment
        save_path = os.path.join(tmp_path, "model.pkl")
        model.save(save_path)
        
        # Load and verify
        loaded_model = ClusteringModel.load(save_path)
        assert loaded_model.n_clusters == 3
        assert loaded_model.cluster_sizes_ == original_sizes
        
        # Ensure predictions are identical
        loaded_predictions = loaded_model.predict(blob_data)
        assert np.array_equal(original_predictions, loaded_predictions)

    def test_predict_raises_if_unfitted(self, blob_data):
        """Test that predict() raises RuntimeError if called before the model is fitted."""
        model = ClusteringModel(n_clusters=3)
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(blob_data)

    def test_fit_auto_finds_k(self, blob_data):
        """Test that fit() automatically finds optimal k if n_clusters is not set."""
        model = ClusteringModel() # n_clusters is None
        model.fit(blob_data) # Should call find_optimal_k
        
        assert model.n_clusters is not None
        assert model.kmeans is not None
        assert len(np.unique(model.predict(blob_data))) == model.n_clusters
