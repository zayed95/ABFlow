import pytest
import uuid
import random
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import func
from api.main import app
from db.session import SessionLocal, engine
from db.models import Base, Experiment, Event, Assignment, SegmentModel, ExperimentStatus
from db.repositories import experiment_repo
from tasks.train_segmentation import train_segmentation_task

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Ensure tables are created before running tests."""
    Base.metadata.create_all(bind=engine)
    yield
    # Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session():
    """Provides a transactional database session for each test."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def seed_user_profiles(db, exp_id, start_idx, n_users, profile_type, ref_time):
    """
    Seeds user event data based on a profile.
    Profiles: 'high-value', 'casual', 'new'
    """
    events = []
    for i in range(start_idx, start_idx + n_users):
        user_id = f"user_{profile_type}_{i}"
        
        if profile_type == 'high-value':
            n_events = 50
            val_range = (80, 120)
            conv_prob = 0.3
        elif profile_type == 'casual':
            n_events = 15
            val_range = (10, 30)
            conv_prob = 0.05
        else: # new
            n_events = 3
            val_range = (1, 5)
            conv_prob = 0.01
            
        for _ in range(n_events):
            occurred_at = ref_time - timedelta(minutes=random.randint(1, 10000))
            
            event = Event(
                experiment_id=exp_id,
                user_id=user_id,
                event_type='conversion' if random.random() < conv_prob else 'view',
                metric_value=random.uniform(*val_range),
                occurred_at=occurred_at
            )
            events.append(event)
            
    db.add_all(events)
    db.commit()

def test_segmentation_flow(db_session):
    """
    Integration test for the full segmentation flow:
    1. Seed pre-experiment data for 600 users.
    2. Train the model.
    3. Enroll users with known profiles.
    4. Verify assignment and clustering accuracy.
    5. Verify cluster interpretability (monetary averages).
    """
    ref_time = datetime.utcnow()
    exp_name = f"Flow_Test_{uuid.uuid4().hex[:6]}"
    exp_data = {
        "name": exp_name,
        "config": {
            "feature_schema": {"event_types": ["view", "conversion"]},
            "min_segment_size": 10
        },
        "seed": 999
    }
    exp = experiment_repo.create_experiment(db_session, exp_data)
    
    seed_user_profiles(db_session, exp.id, 0, 200, 'high-value', ref_time)
    seed_user_profiles(db_session, exp.id, 200, 200, 'casual', ref_time)
    seed_user_profiles(db_session, exp.id, 400, 200, 'new', ref_time)
    
    experiment_repo.update_status(db_session, exp.id, ExperimentStatus.running)
    train_segmentation_task.apply(args=[str(exp.id)]).get()
    
    seed_user_profiles(db_session, exp.id, 1000, 100, 'high-value', ref_time)
    seed_user_profiles(db_session, exp.id, 1100, 100, 'casual', ref_time)
    
    results = []
    for i in range(1000, 1100):
        user_id = f"user_high-value_{i}"
        response = client.post("/assignments/enroll", json={"experiment_id": str(exp.id), "user_id": user_id})
        data = response.json()
        assert data["segment_id"] is not None
        results.append({"user_id": user_id, "profile": "high-value", "segment_id": data["segment_id"]})

    for i in range(1100, 1200):
        user_id = f"user_casual_{i}"
        response = client.post("/assignments/enroll", json={"experiment_id": str(exp.id), "user_id": user_id})
        data = response.json()
        assert data["segment_id"] is not None
        results.append({"user_id": user_id, "profile": "casual", "segment_id": data["segment_id"]})

    high_value_segs = [r["segment_id"] for r in results if r["profile"] == "high-value"]
    consensus_segment = max(set(high_value_segs), key=high_value_segs.count)
    consensus_pct = high_value_segs.count(consensus_segment) / len(high_value_segs)
    assert consensus_pct >= 0.80

    # 6. Verify cluster interpretability: compute average monetary_sum per segment
    segment_stats = {} # segment_id -> list of total_monetary
    for r in results:
        user_id = r["user_id"]
        seg_id = r["segment_id"]
        
        # Calculate total monetary for this user from historical events
        total_val = db_session.query(func.sum(Event.metric_value)).filter(
            Event.user_id == user_id,
            Event.experiment_id == exp.id
        ).scalar() or 0.0
        
        if seg_id not in segment_stats:
            segment_stats[seg_id] = []
        segment_stats[seg_id].append(total_val)
        
    avg_monetary = {seg_id: sum(vals)/len(vals) for seg_id, vals in segment_stats.items()}
    
    # High-value cluster average
    high_val_avg = avg_monetary[consensus_segment]
    
    # Compare with other segments (casual, etc)
    other_avgs = [avg for seg_id, avg in avg_monetary.items() if seg_id != consensus_segment]
    
    print(f"High-value segment ({consensus_segment}) average monetary: {high_val_avg:.2f}")
    for i, avg in enumerate(other_avgs):
        print(f"Other segment average monetary: {avg:.2f}")
        # High-value should be significantly higher (e.g., > 3x casual)
        assert high_val_avg > avg * 3, f"High-value segment ({high_val_avg}) is not significantly higher than other segment ({avg})"

def test_enrollment_idempotency(db_session):
    """
    Test that enrolling the same user twice returns the same assignment and segment_id.
    """
    exp_data = {
        "name": f"Idempotency_Test_{uuid.uuid4().hex[:6]}",
        "config": {"feature_schema": {}},
        "seed": 123
    }
    exp = experiment_repo.create_experiment(db_session, exp_data)
    experiment_repo.update_status(db_session, exp.id, ExperimentStatus.running)
    
    user_id = "test_idempotent_user"
    resp1 = client.post("/assignments/enroll", json={"experiment_id": str(exp.id), "user_id": user_id})
    data1 = resp1.json()
    
    resp2 = client.post("/assignments/enroll", json={"experiment_id": str(exp.id), "user_id": user_id})
    data2 = resp2.json()
    
    assert data1["id"] == data2["id"]
    assert data1["segment_id"] == data2["segment_id"]

def test_frozen_segment_assignment(db_session):
    """
    Test that once a segment_id is set:
    1. It is returned identically upon re-enrollment, even if historical event data changes.
    2. The application-level guard prevents manual updates in the DB.
    """
    ref_time = datetime.utcnow()
    exp_data = {
        "name": f"Frozen_Test_{uuid.uuid4().hex[:6]}",
        "config": {"feature_schema": {"event_types": ["view"]}},
        "seed": 777
    }
    exp = experiment_repo.create_experiment(db_session, exp_data)
    
    seed_user_profiles(db_session, exp.id, 0, 600, 'casual', ref_time)
    experiment_repo.update_status(db_session, exp.id, ExperimentStatus.running)
    train_segmentation_task.apply(args=[str(exp.id)]).get()
    
    user_id = "user_casual_5000"
    seed_user_profiles(db_session, exp.id, 5000, 1, 'casual', ref_time)
    
    resp1 = client.post("/assignments/enroll", json={"experiment_id": str(exp.id), "user_id": user_id})
    data1 = resp1.json()
    seg_id_initial = data1["segment_id"]
    
    db_session.query(Event).filter(Event.user_id == user_id).update({"metric_value": 10000.0})
    db_session.commit()
    
    resp2 = client.post("/assignments/enroll", json={"experiment_id": str(exp.id), "user_id": user_id})
    data2 = resp2.json()
    assert data2["segment_id"] == seg_id_initial
    
    assignment = db_session.query(Assignment).filter(
        Assignment.experiment_id == exp.id,
        Assignment.user_id == user_id
    ).first()
    
    with pytest.raises(ValueError, match="segment_id is frozen"):
        assignment.segment_id = 999
        db_session.commit()
