import pytest
import uuid
import random
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from api.main import app
from db.models import Experiment, Event, Assignment, ExperimentStatus
from db.repositories import experiment_repo, posterior_repo
from tasks.train_segmentation import train_segmentation_task

client = TestClient(app)

def seed_historical_data(db, exp_id, n_users_per_seg, ref_time):
    """
    Seeds historical data for 3 distinct user segments.
    """
    events = []
    for seg_idx in range(3):
        for i in range(n_users_per_seg):
            user_id = f"user_profile_{seg_idx}_{i}"
            
            if seg_idx == 1: # High-value: High frequency, High monetary
                n_events = 30
                val = 150
            elif seg_idx == 0: # Casual: Medium frequency, Medium monetary
                n_events = 10
                val = 50
            else: # New: Low frequency, Low monetary
                n_events = 2
                val = 10
                
            for _ in range(n_events):
                events.append(Event(
                    experiment_id=exp_id,
                    user_id=user_id,
                    event_type="view",
                    metric_value=float(val),
                    occurred_at=ref_time - timedelta(days=random.randint(1, 15))
                ))
    db.add_all(events)
    db.commit()

def test_hte_full_flow(db_session):
    """
    End-to-end HTE flow test:
    1. Seed historical data for clustering (1200 users).
    2. Train model and start experiment.
    3. Enroll users into experiment variants (1200 users total).
    4. Inject 1,200 events with variant-specific CR:
       - Segment 1: Treatment=0.20, Control=0.10 (Strong Uplift)
       - Others: Treatment=0.10, Control=0.10 (Null)
    5. Verify significance in API response after BH correction.
    """
    random.seed(42)
    ref_time = datetime.utcnow()
    exp_id = uuid.uuid4()
    
    # 1. Create Experiment
    exp = experiment_repo.create_experiment(db_session, {
        "id": exp_id,
        "name": f"HTE_E2E_{exp_id.hex[:6]}",
        "config": {"min_samples": 10},
        "seed": 42
    })
    
    # 2. Seed Historical Data (1200 users total)
    seed_historical_data(db_session, exp_id, 400, ref_time)
    
    # 3. Train Segmentation
    experiment_repo.update_status(db_session, exp_id, ExperimentStatus.running)
    train_segmentation_task.apply(args=[str(exp_id)]).get()
    
    # 4. Enroll Users and Inject post-enrollment Events
    for seg_idx in range(3):
        for i in range(400):
            user_id = f"user_profile_{seg_idx}_{i}"
            
            # Enroll user
            response = client.post("/assignments/enroll", json={
                "experiment_id": str(exp_id),
                "user_id": user_id
            })
            assert response.status_code == 201
            enrollment = response.json()
            variant = enrollment["variant"]
            
            # Logic:
            # Segment 1 gets CR=0.20 in Treatment, CR=0.10 in Control
            # Others get CR=0.10 in both
            if seg_idx == 1:
                cr = 0.20 if variant == "treatment" else 0.10
            else:
                cr = 0.10
            
            # Add 1 event per user
            is_conv = random.random() < cr
            db_session.add(Event(
                experiment_id=exp_id,
                user_id=user_id,
                event_type="conversion" if is_conv else "view",
                metric_value=1.0 if is_conv else 0.0,
                occurred_at=ref_time + timedelta(minutes=random.randint(1, 100))
            ))
    db_session.commit()

    # Add dummy snapshots for overall Bayesian results
    posterior_repo.save_snapshot(
        db_session, exp_id, "control", alpha_post=1, beta_post=1, n_trials=100, n_conversions=10, last_processed_at=datetime.utcnow()
    )
    posterior_repo.save_snapshot(
        db_session, exp_id, "treatment", alpha_post=1, beta_post=1, n_trials=100, n_conversions=10, last_processed_at=datetime.utcnow()
    )

    # 5. Verify Results via API
    response = client.get(f"/experiments/{exp_id}/results")
    assert response.status_code == 200
    data = response.json()
    
    seg_results = data["segment_results"]
    assert len(seg_results) == 3
    
    # Significant segments (should be only 1: Segment 1)
    significant_segments = [s for s in seg_results if s["significant"]]
    assert len(significant_segments) == 1
    
    high_value_result = significant_segments[0]
    # Check that it's the one with high uplift
    assert high_value_result["delta"] > 0.05
    assert high_value_result["cr_treatment"] > 0.15
    
    # Verify others are not significant
    null_results = [s for s in seg_results if not s["significant"]]
    assert len(null_results) == 2
