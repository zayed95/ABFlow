import numpy as np
from typing import Dict, Any
from core.segmentation.clustering import ClusteringModel
from core.segmentation.features import FeatureExtractor
from db.models import SegmentModel

def assign_segment(user_features: Dict[str, Any], segment_model: SegmentModel) -> int:
    """
    Assigns a segment to a user based on their features and a trained SegmentModel.
    Implements an imputation strategy: missing features are filled with the mean values 
    from the training set (stored in the scaler).
    
    Args:
        user_features: A dictionary containing the user's features.
        segment_model: The SegmentModel database instance containing the trained artifacts.
        
    Returns:
        int: The assigned segment ID (cluster index).
    """
    # 1. Load artifacts
    # The ClusteringModel.load_artifacts method returns both the model and the fitted extractor.
    model, extractor = ClusteringModel.load_artifacts(segment_model.kmeans_artifact)
    
    if not hasattr(extractor, "scaler"):
        raise RuntimeError("The reconstructed FeatureExtractor does not contain a fitted scaler.")

    # 2. Validation and Imputation against feature names
    # FeatureExtractor.get_feature_names() returns ['recency', 'frequency', 'monetary', 'session_depth', 'conversion_rate']
    feature_names = extractor.get_feature_names()
    final_features = []
    
    for i, name in enumerate(feature_names):
        # Check if feature exists in the user input and is not None
        if name in user_features and user_features[name] is not None:
            final_features.append(user_features[name])
        else:
            # Imputation strategy: Use the mean value from the training set.
            # This is stored in extractor.scaler.mean_[index].
            # This ensures that missing data defaults to the 'average' behavior,
            # which results in a scaled value of 0.0.
            mean_val = extractor.scaler.mean_[i]
            final_features.append(mean_val)
            
    # 3. Transform the user's features using the frozen scaler
    X = np.array([final_features])
    X_scaled = extractor.scaler.transform(X)
    
    # 4. Call model.predict() to get the cluster assignment
    preds = model.predict(X_scaled)
    
    # Return the prediction as an integer
    return int(preds[0])
