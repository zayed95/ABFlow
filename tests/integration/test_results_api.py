import pytest
import uuid
from fastapi.testclient import TestClient
from api.main import app
from db.models import Experiment, Event, Assignment
from db.repositories import posterior_repo
from datetime import datetime

client = TestClient(app)

def test_hte_results_significance(db_session):
    """
    Test that the results endpoint correctly identifies significant segments.
    3 segments:
    - Seg 0: Null (100 users each, 10 conv each -> 10% CR)
    - Seg 1: Positive (100 users each, 10 control conv, 30 treatment conv -> 10% vs 30% CR)
    - Seg 2: Null (100 users each, 10 conv each -> 10% CR)
    """
    exp_id = uuid.uuid4()
    exp_name = f"HTE_Test_{exp_id.hex[:6]}"
    exp_data = {
        "id": exp_id,
        "name": exp_name,
        "config": {"prior_type": "uniform", "min_samples": 10},
        "seed": 42
    }
    exp = Experiment(**exp_data)
    db_session.add(exp)
    db_session.commit()

    # Seed Assignments and Events
    variants = ["control", "treatment"]
    for seg_id in range(3):
        for variant in variants:
            # 100 users per (seg, variant) to pass the 30-user guard
            for i in range(100):
                user_id = f"user_{seg_id}_{variant}_{i}"
                db_session.add(Assignment(
                    experiment_id=exp_id,
                    user_id=user_id,
                    variant=variant,
                    segment_id=seg_id
                ))
                
                # Conversions
                # Seg 1 Treatment has 30 conversions (30% CR), others have 10 (10% CR)
                is_conv = False
                if seg_id == 1 and variant == "treatment":
                    if i < 30: is_conv = True
                else:
                    if i < 10: is_conv = True
                
                if is_conv:
                    db_session.add(Event(
                        experiment_id=exp_id,
                        user_id=user_id,
                        event_type="conversion",
                        metric_value=1.0
                    ))
    db_session.commit()

    # Add dummy Posterior Snapshots (required for the endpoint)
    now = datetime.utcnow()
    posterior_repo.save_snapshot(
        db_session, exp_id, "control", alpha_post=11.0, beta_post=91.0, 
        n_trials=100, n_conversions=10, last_processed_at=now
    )
    posterior_repo.save_snapshot(
        db_session, exp_id, "treatment", alpha_post=11.0, beta_post=91.0, 
        n_trials=100, n_conversions=10, last_processed_at=now
    )

    # Call Endpoint
    response = client.get(f"/experiments/{exp_id}/results")
    assert response.status_code == 200
    data = response.json()
    
    # Assertions
    assert data["n_segments"] == 3
    # Only segment 1 should be significant (0.1 vs 0.3)
    # Segments 0 and 2 are 0.1 vs 0.1
    assert data["n_significant_segments"] == 1
    
    # Check specific segment results
    seg_results = data["segment_results"]
    
    # Segment 1 should be significant
    seg_1 = next(r for r in seg_results if r["segment_id"] == 1)
    assert seg_1["significant"] is True
    assert seg_1["cr_control"] == 0.1
    assert seg_1["cr_treatment"] == 0.3
    
    # Segment 0 and 2 should not be significant
    seg_0 = next(r for r in seg_results if r["segment_id"] == 0)
    assert seg_0["significant"] is False
    assert seg_0["cr_control"] == 0.1
    assert seg_0["cr_treatment"] == 0.1

    seg_2 = next(r for r in seg_results if r["segment_id"] == 2)
    assert seg_2["significant"] is False
    assert seg_2["cr_control"] == 0.1
    assert seg_2["cr_treatment"] == 0.1
