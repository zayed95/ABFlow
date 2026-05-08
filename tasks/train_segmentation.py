import uuid
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from celery import shared_task
from db.session import SessionLocal
from db.models import Experiment, Event, SegmentModel
from db.repositories import experiment_repo
from core.segmentation.features import FeatureExtractor
from core.segmentation.clustering import ClusteringModel

logger = logging.getLogger(__name__)

class TaskError(Exception):
    """Exception raised for errors in the segmentation training task."""
    pass

@shared_task
def train_segmentation_task(experiment_id: str):
    """
    Segmentation training task following specific business rules:
    1. Load experiment config and feature_schema.
    2. Filter for pre-experiment events only.
    3. Enforce a minimum of 500 users for statistical significance.
    4. Fit feature extraction and clustering pipelines.
    5. Ensure every cluster meets the min_segment_size requirement (retrain if not).
    6. Persist artifacts to DB and log metadata to experiment config.
    """
    db = SessionLocal()
    try:
        exp_uuid = uuid.UUID(experiment_id)
        
        # (1) Load the experiment config from DB
        experiment = experiment_repo.get_experiment(db, exp_uuid)
        if not experiment:
            raise TaskError(f"Experiment {experiment_id} not found.")

        config = experiment.config
        feature_schema = config.get("feature_schema", {})
        allowed_event_types = feature_schema.get("event_types", [])
        min_segment_size = config.get("min_segment_size", 30)

        # (2) Query the Event table for pre-experiment data only
        query = db.query(Event).filter(
            Event.experiment_id == exp_uuid,
            Event.occurred_at < experiment.created_at
        )
        
        if allowed_event_types:
            query = query.filter(Event.event_type.in_(allowed_event_types))
            
        events = query.all()

        # (3) Data Volume Check
        unique_users = {e.user_id for e in events}
        if len(unique_users) < 500:
            raise TaskError(
                f"Insufficient pre-experiment data: only {len(unique_users)} users found. "
                "The model will not be meaningful with less than 500 users."
            )

        # Prepare DataFrame for pipeline
        df_events = pd.DataFrame([{
            'user_id': e.user_id,
            'occurred_at': e.occurred_at,
            'metric_value': e.metric_value,
            'event_type': e.event_type,
            'session_id': getattr(e, 'session_id', None)
        } for e in events])

        # (4) Feature Extraction Pipeline
        extractor = FeatureExtractor(feature_schema=feature_schema)
        X_scaled = extractor.fit_transform(df_events)
        
        # (5) Clustering Model Training with min_segment_size constraint
        model = ClusteringModel()
        model.find_optimal_k(X_scaled)
        
        while model.n_clusters > 1:
            model.fit(X_scaled)
            summary = model.cluster_summary()
            min_actual_size = summary['size'].min()
            
            if min_actual_size >= min_segment_size:
                logger.info(f"Final model chosen with K={model.n_clusters} (min cluster size: {min_actual_size})")
                break
            
            logger.warning(
                f"Cluster size constraint violated: smallest cluster has {min_actual_size} users, "
                f"required {min_segment_size}. Reducing K from {model.n_clusters} to {model.n_clusters - 1}."
            )
            model.n_clusters -= 1
            
        if model.n_clusters < 2:
            logger.warning("Could not find a clustering solution that satisfies min_segment_size.")

        # (6) Save Artifacts and Metadata
        artifact_bytes = model.save_artifacts(extractor)
        
        # Prepare metadata for logging
        feature_names = extractor.get_feature_names() if hasattr(extractor, 'get_feature_names') else None
        summary_df = model.cluster_summary(feature_names=feature_names)
        
        # Convert summary to a JSON-serializable list of dicts
        cluster_meta = summary_df.to_dict(orient='records')
        
        segmentation_meta = {
            'clusters': cluster_meta,
            'silhouette_score': float(model.silhouette_score_),
            'n_clusters': model.n_clusters,
            'trained_at': datetime.utcnow().isoformat()
        }
        
        # Update Experiment config JSONB field
        # Re-assign to trigger SQLAlchemy change detection
        new_config = dict(experiment.config)
        new_config['segmentation_meta'] = segmentation_meta
        experiment.config = new_config
        
        # Create SegmentModel record
        new_segment_model = SegmentModel(
            experiment_id=exp_uuid,
            model_version=datetime.utcnow().strftime("%Y%m%d%H%M%S"),
            scaler_artifact=b"", 
            kmeans_artifact=artifact_bytes,
            n_clusters=model.n_clusters
        )
        db.add(new_segment_model)
        db.commit()
        
        logger.info(f"Successfully trained segmentation for {experiment_id}. Metadata logged to config.")

    except TaskError as te:
        logger.error(f"Task aborted: {str(te)}")
        raise te
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error in train_segmentation_task: {str(e)}")
        raise
    finally:
        db.close()
