from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid
from typing import List
from db.session import get_db
from api.schemas import ExperimentCreate, ExperimentResponse, ResultsResponse, SegmentProfile
from db.repositories import experiment_repo
from services import experiment_service

experiment_router = APIRouter(prefix="/experiments", tags=["experiments"])

@experiment_router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_new_experiment(experiment: ExperimentCreate, db: Session = Depends(get_db)):
    """
    Creates a new experiment.
    """
    # Convert Pydantic model to dict for the repo
    experiment_data = experiment.model_dump()
    return experiment_repo.create_experiment(db, experiment_data)

@experiment_router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment_details(experiment_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Retrieves details of a specific experiment.
    Returns 404 if the experiment does not exist.
    """
    db_experiment = experiment_repo.get_experiment(db, experiment_id)
    if db_experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    return db_experiment

@experiment_router.post("/{experiment_id}/start", response_model=ExperimentResponse)
def start_experiment_endpoint(experiment_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Triggers the start of an experiment and its segmentation training.
    """
    db_experiment = experiment_service.start_experiment(db, str(experiment_id))
    if db_experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    return db_experiment

@experiment_router.get("/{experiment_id}/results", response_model=ResultsResponse)
def get_experiment_results(experiment_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Retrieves the analysis results for an experiment, including:
    1. Overall Bayesian posterior statistics.
    2. Per-segment Heterogeneous Treatment Effect (HTE) analysis with BH correction.
    """
    from collections import defaultdict
    from db.repositories import posterior_repo, event_repo
    from core.segmentation import hte
    from core.sequential.bayesian import BetaBinomialPosterior
    from core.sequential.decision import evaluate_decision

    # 1. Fetch Experiment
    experiment = experiment_repo.get_experiment(db, experiment_id)
    if not experiment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    
    # 2. Build Overall Posterior Response
    control_snap = posterior_repo.get_latest_snapshot(db, experiment_id, "control")
    treatment_snap = posterior_repo.get_latest_snapshot(db, experiment_id, "treatment")
    
    overall_results = None
    if control_snap and treatment_snap:
        config = experiment.config
        
        # Hydrate Posteriors
        post_a = BetaBinomialPosterior(
            prior_type=config.get("prior_type", "uniform"),
            historical_cr=config.get("historical_cr"),
            alpha_prior=config.get("alpha_prior", 1.0),
            beta_prior=config.get("beta_prior", 1.0)
        )
        post_a.alpha_posterior = control_snap.alpha_post
        post_a.beta_posterior = control_snap.beta_post
        
        post_b = BetaBinomialPosterior(
            prior_type=config.get("prior_type", "uniform"),
            historical_cr=config.get("historical_cr"),
            alpha_prior=config.get("alpha_prior", 1.0),
            beta_prior=config.get("beta_prior", 1.0)
        )
        post_b.alpha_posterior = treatment_snap.alpha_post
        post_b.beta_posterior = treatment_snap.beta_post
        
        decision_res = evaluate_decision(
            post_a, post_b,
            threshold_win=config.get("threshold_win", 0.95),
            threshold_null=config.get("threshold_null", 0.05),
            min_samples=config.get("min_samples", 100)
        )
        
        overall_results = {
            "control": {
                "alpha_posterior": post_a.alpha_posterior,
                "beta_posterior": post_a.beta_posterior,
                "expected_value": post_a.expected_value,
                "variance": post_a.variance
            },
            "treatment": {
                "alpha_posterior": post_b.alpha_posterior,
                "beta_posterior": post_b.beta_posterior,
                "expected_value": post_b.expected_value,
                "variance": post_b.variance
            },
            "prob_b_beats_a": decision_res.prob_b_beats_a,
            "decision": decision_res.decision.value
        }

    # 3. Fetch Per-Segment Metrics
    segment_metrics = event_repo.get_segment_metrics(db, experiment_id)
    
    # Identify excluded segments for warnings
    from db.models import Assignment
    all_assigned_segments = [
        r[0] for r in db.query(Assignment.segment_id)
        .filter(Assignment.experiment_id == experiment_id)
        .distinct().all() 
        if r[0] is not None
    ]
    included_segments = {m.segment_id for m in segment_metrics}
    
    warnings = []
    for seg_id in all_assigned_segments:
        if seg_id not in included_segments:
            warnings.append(
                f"Segment {seg_id} excluded from HTE analysis due to insufficient sample size (< 30 users per variant)."
            )

    # 4. Compute HTE for each segment
    segment_results = []
    metrics_by_segment = defaultdict(dict)
    for m in segment_metrics:
        metrics_by_segment[m.segment_id][m.variant] = m
    
    for seg_id, variants in metrics_by_segment.items():
        if "control" in variants and "treatment" in variants:
            # Only run test if both variants exist and passed the repo guards (30+ users)
            res = hte.run_segment_test(variants["control"], variants["treatment"])
            segment_results.append(res)
    
    # 5. Apply Benjamini-Hochberg Correction for multiple testing
    if segment_results:
        segment_results = hte.apply_bh_correction(segment_results)
    
    n_significant = sum(1 for r in segment_results if r.significant)
    
    return {
        "overall": overall_results,
        "segment_results": segment_results,
        "n_segments": len(segment_results),
        "n_significant_segments": n_significant,
        "warnings": warnings
    }

@experiment_router.get("/{experiment_id}/segments", response_model=List[SegmentProfile])
def get_segment_profiles(experiment_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Returns a summary of segment profiles for an experiment.
    Labels segments as 'high_value', 'new_user', or 'casual' based on centroid features.
    """
    from db.repositories import assignment_repo
    from core.segmentation import ClusteringModel
    import pandas as pd
    import numpy as np

    # 1. Fetch Segment Model
    model_record = assignment_repo.get_segment_model(db, experiment_id)
    if not model_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="No segment model found for this experiment. Training might be in progress."
        )

    # 2. Load Artifacts
    clustering_model, extractor = ClusteringModel.load_artifacts(model_record.kmeans_artifact)
    
    # 3. Extract Centroids and Inverse Transform
    scaled_centroids = clustering_model.kmeans.cluster_centers_
    feature_names = extractor.get_feature_names()
    
    # Use the fitted scaler from the extractor to return raw feature values
    if hasattr(extractor, 'scaler') and extractor.scaler is not None:
        raw_centroids = extractor.scaler.inverse_transform(scaled_centroids)
    else:
        raw_centroids = scaled_centroids
        
    df = pd.DataFrame(raw_centroids, columns=feature_names)
    
    # 4. Auto-Labeling Logic
    labels = ClusteringModel.label_clusters(df)
    
    profiles = []
    for i in range(len(df)):
        profiles.append({
            "segment_id": i,
            "label": labels.get(i, "casual"),
            "centroids": df.iloc[i].to_dict()
        })
        
    return profiles
