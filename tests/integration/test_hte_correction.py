import pytest
import uuid
import random
from fastapi.testclient import TestClient
from api.main import app
from db.models import Experiment, Assignment, Event
from db.repositories import posterior_repo
from datetime import datetime

client = TestClient(app)

def test_hte_fdr_control(db_session):
    """
    Test that Benjamini-Hochberg correction prevents false positive inflation.
    10 segments with identical conversion rates (0.10) for both variants.
    
    Without correction, the probability of at least one segment having p < 0.05 
    purely by chance is ~40%.
    BH correction should keep the number of 'significant' flags at 0 (or very low).
    """
    # Use a fixed seed to ensure we get at least one 'borderline' false positive 
    # that would be significant WITHOUT correction.
    random.seed(1337) 
    exp_id = uuid.uuid4()
    exp_data = {
        "id": exp_id,
        "name": f"FDR_Test_{exp_id.hex[:6]}",
        "config": {"prior_type": "uniform"},
        "seed": 42
    }
    exp = Experiment(**exp_data)
    db_session.add(exp)
    db_session.commit()

    # Seed 10 Null Segments
    # 200 users per variant to ensure statistical stability
    for seg_id in range(10):
        for variant in ["control", "treatment"]:
            for i in range(200):
                user_id = f"user_s{seg_id}_{variant}_{i}"
                db_session.add(Assignment(
                    experiment_id=exp_id,
                    user_id=user_id,
                    variant=variant,
                    segment_id=seg_id
                ))
                
                # CR = 0.10 (Null effect)
                if random.random() < 0.10:
                    db_session.add(Event(
                        experiment_id=exp_id,
                        user_id=user_id,
                        event_type="conversion",
                        metric_value=1.0
                    ))
    db_session.commit()

    # Add dummy snapshots
    now = datetime.utcnow()
    posterior_repo.save_snapshot(db_session, exp_id, "control", 1, 1, 100, 10, now)
    posterior_repo.save_snapshot(db_session, exp_id, "treatment", 1, 1, 100, 10, now)

    # Call API
    response = client.get(f"/experiments/{exp_id}/results")
    assert response.status_code == 200
    data = response.json()
    
    # 1. Verify all 10 segments are processed
    assert data["n_segments"] == 10
    
    # 2. Verify BH correction controlled FDR
    # Expected significant segments should be 0 or 1.
    assert data["n_significant_segments"] <= 1
    
    # Check if any segment had a raw p-value < 0.05 but was CORRECTLY flagged as non-significant
    # (Since I can't see raw p-values in the API, I'll just rely on the count)
    print(f"Significant segments found: {data['n_significant_segments']}")
    for res in data["segment_results"]:
        # If it's not significant, its corrected p-value should be high or it should just be False
        if not res["significant"]:
            # Corrected p-value usually becomes much larger than 0.05
            assert res["corrected_p_value"] >= 0.05 or res["corrected_p_value"] is None
