import pytest
import uuid
import random
from fastapi.testclient import TestClient
from api.main import app
from db.models import Experiment, Assignment, Event
from db.repositories import posterior_repo
from datetime import datetime

client = TestClient(app)

@pytest.fixture
def benchmark_exp(db_session):
    """
    Seeds a benchmark experiment with 10 segments and 10,000 total events.
    Each user has 1 event.
    """
    exp_id = uuid.uuid4()
    exp = Experiment(
        id=exp_id,
        name=f"Performance_Bench_{exp_id.hex[:6]}",
        config={"prior_type": "uniform"},
        seed=101
    )
    db_session.add(exp)
    
    # 10 segments, 1,000 users each = 10,000 users/events
    for seg_id in range(10):
        for i in range(1000):
            user_id = f"user_{seg_id}_{i}"
            variant = "control" if i % 2 == 0 else "treatment"
            db_session.add(Assignment(
                experiment_id=exp_id,
                user_id=user_id,
                variant=variant,
                segment_id=seg_id
            ))
            
            # One event per user
            db_session.add(Event(
                experiment_id=exp_id,
                user_id=user_id,
                event_type="conversion" if random.random() < 0.1 else "view",
                metric_value=1.0
            ))
            
        # Commit in batches per segment for speed
        db_session.commit()
    
    # Add dummy snapshots
    now = datetime.utcnow()
    posterior_repo.save_snapshot(db_session, exp_id, "control", 1, 1, 500, 50, now)
    posterior_repo.save_snapshot(db_session, exp_id, "treatment", 1, 1, 500, 50, now)
    
    return exp_id

def test_results_endpoint_performance(benchmark, benchmark_exp):
    """
    Benchmark the /results endpoint with 10,000 events.
    Target: Mean latency < 500ms.
    """
    def call_results():
        response = client.get(f"/experiments/{benchmark_exp}/results")
        assert response.status_code == 200
        return response

    # Run the benchmark
    response = benchmark(call_results)
    
    # Assertions on performance
    # stats.mean is in seconds
    mean_latency_ms = benchmark.stats.stats.mean * 1000
    print(f"\nMean Latency: {mean_latency_ms:.2f}ms")
    
    assert response.status_code == 200
    assert benchmark.stats.stats.mean < 0.500, f"Performance target failed: {mean_latency_ms:.2f}ms > 500ms"
