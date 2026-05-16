import pytest
import uuid
from fastapi.testclient import TestClient
from api.main import app
from db.models import Experiment, Assignment
from db.repositories import posterior_repo
from datetime import datetime

client = TestClient(app)

def test_hte_min_size_guard(db_session):
    """
    Test the minimum segment size guard:
    - Segment 0: 35 users per variant (Should be included, 35 >= 30)
    - Segment 1: 15 users per variant (Should be excluded, 15 < 30)
    
    Verify that Segment 1 is excluded from results and a warning is returned.
    """
    exp_id = uuid.uuid4()
    exp_data = {
        "id": exp_id,
        "name": f"Guard_Test_{exp_id.hex[:6]}",
        "config": {"prior_type": "uniform"},
        "seed": 101
    }
    exp = Experiment(**exp_data)
    db_session.add(exp)
    db_session.commit()

    # Seed Assignments
    # Segment 0: Large enough (35 per variant)
    for variant in ["control", "treatment"]:
        for i in range(35):
            db_session.add(Assignment(
                experiment_id=exp_id,
                user_id=f"user_s0_{variant}_{i}",
                variant=variant,
                segment_id=0
            ))
            
    # Segment 1: Too small (15 per variant)
    for variant in ["control", "treatment"]:
        for i in range(15):
            db_session.add(Assignment(
                experiment_id=exp_id,
                user_id=f"user_s1_{variant}_{i}",
                variant=variant,
                segment_id=1
            ))
    db_session.commit()

    # Add dummy snapshots to allow the results endpoint to function
    now = datetime.utcnow()
    posterior_repo.save_snapshot(
        db_session, exp_id, "control", alpha_post=1.0, beta_post=1.0, 
        n_trials=100, n_conversions=10, last_processed_at=now
    )
    posterior_repo.save_snapshot(
        db_session, exp_id, "treatment", alpha_post=1.0, beta_post=1.0, 
        n_trials=100, n_conversions=10, last_processed_at=now
    )

    # Call Results Endpoint
    response = client.get(f"/experiments/{exp_id}/results")
    assert response.status_code == 200
    data = response.json()
    
    # 1. Check Included Segments
    seg_results = data["segment_results"]
    included_ids = [r["segment_id"] for r in seg_results]
    
    # Segment 0 should be present
    assert 0 in included_ids
    # Segment 1 should be ABSENT
    assert 1 not in included_ids
    
    # 2. Check Warnings
    warnings = data["warnings"]
    assert len(warnings) >= 1
    assert any("Segment 1 excluded" in w for w in warnings)
    assert any("insufficient sample size" in w for w in warnings)
    
    print(f"Warnings received: {warnings}")
