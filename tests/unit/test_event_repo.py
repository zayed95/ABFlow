import pytest
import uuid
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base, Assignment, Event, Experiment, ExperimentStatus
from db.repositories import event_repo

# Setup in-memory SQLite for unit tests
engine = create_engine("sqlite:///:memory:")
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db_session():
    """Provides an in-memory SQLite session for testing."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

def test_get_segment_metrics_fixed_fixture(db_session):
    """
    Verify that get_segment_metrics returns exactly 6 rows for a fixture with 
    3 segments and 2 variants each, all meeting the minimum size guard (30).
    """
    # 1. Setup Experiment
    exp_id = uuid.uuid4()
    exp = Experiment(
        id=exp_id,
        name="Unit_Test_HTE",
        status=ExperimentStatus.running,
        config={"feature_schema": {}},
        seed=42
    )
    db_session.add(exp)
    db_session.commit()

    # 2. Define Fixture Data
    # Each variant has 30 users to pass the MIN_USERS = 30 guard.
    segment_data = [
        # (segment_id, variant, n_users, n_conversions)
        (1, "control", 30, 3),    # CR: 0.1
        (1, "treatment", 30, 6),  # CR: 0.2
        (2, "control", 30, 0),    # CR: 0.0
        (2, "treatment", 30, 15), # CR: 0.5
        (3, "control", 40, 10),   # CR: 0.25
        (3, "treatment", 50, 25), # CR: 0.5
    ]

    for seg_id, variant, n_users, n_convs in segment_data:
        for i in range(n_users):
            user_id = f"user_{seg_id}_{variant}_{i}"
            assignment = Assignment(
                experiment_id=exp_id, 
                user_id=user_id, 
                variant=variant, 
                segment_id=seg_id
            )
            db_session.add(assignment)
            
            # Add conversion events for a subset of users
            if i < n_convs:
                event = Event(
                    experiment_id=exp_id, 
                    user_id=user_id, 
                    event_type="conversion", 
                    metric_value=1.0
                )
                db_session.add(event)
    
    db_session.commit()

    # 3. Execute the repository method
    metrics = event_repo.get_segment_metrics(db_session, exp_id)

    # 4. Verify results
    assert len(metrics) == 6, f"Expected 6 metric rows, got {len(metrics)}"
    
    # Sort for deterministic assertions
    metrics.sort(key=lambda x: (x.segment_id, x.variant))
    
    # Verify Segment 1
    assert metrics[0].segment_id == 1 and metrics[0].variant == "control"
    assert metrics[0].n_users == 30 and metrics[0].n_conversions == 3
    assert metrics[0].conversion_rate == pytest.approx(0.1)
    assert metrics[0].is_testable is True

    assert metrics[1].segment_id == 1 and metrics[1].variant == "treatment"
    assert metrics[1].n_users == 30 and metrics[1].n_conversions == 6
    assert metrics[1].conversion_rate == pytest.approx(0.2)
    assert metrics[1].is_testable is True

    # Verify Segment 2
    assert metrics[2].segment_id == 2 and metrics[2].variant == "control"
    assert metrics[2].n_users == 30 and metrics[2].n_conversions == 0
    assert metrics[2].is_testable is True

    assert metrics[3].segment_id == 2 and metrics[3].variant == "treatment"
    assert metrics[3].n_users == 30 and metrics[3].n_conversions == 15
    assert metrics[3].conversion_rate == pytest.approx(0.5)

    # Verify Segment 3
    assert metrics[4].segment_id == 3 and metrics[4].variant == "control"
    assert metrics[4].n_users == 40 and metrics[4].n_conversions == 10
    
    assert metrics[5].segment_id == 3 and metrics[5].variant == "treatment"
    assert metrics[5].n_users == 50 and metrics[5].n_conversions == 25
    assert metrics[5].conversion_rate == pytest.approx(0.5)

def test_get_segment_metrics_exclusion_guard(db_session):
    """
    Verify that segments with < 30 users in any variant are excluded.
    """
    exp_id = uuid.uuid4()
    exp = Experiment(id=exp_id, name="Guard_Test", status=ExperimentStatus.running, config={}, seed=1)
    db_session.add(exp)
    
    # Segment 1: control=30, treatment=29 (SHOULD BE EXCLUDED)
    for i in range(30):
        db_session.add(Assignment(experiment_id=exp_id, user_id=f"u1c{i}", variant="control", segment_id=1))
    for i in range(29):
        db_session.add(Assignment(experiment_id=exp_id, user_id=f"u1t{i}", variant="treatment", segment_id=1))

    # Segment 2: control=30, treatment=30 (SHOULD BE INCLUDED)
    for i in range(30):
        db_session.add(Assignment(experiment_id=exp_id, user_id=f"u2c{i}", variant="control", segment_id=2))
        db_session.add(Assignment(experiment_id=exp_id, user_id=f"u2t{i}", variant="treatment", segment_id=2))

    db_session.commit()

    metrics = event_repo.get_segment_metrics(db_session, exp_id)
    
    # Only Segment 2 should be present (2 rows: control and treatment)
    assert len(metrics) == 2
    assert all(m.segment_id == 2 for m in metrics)
