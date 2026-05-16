import pytest
import uuid
from datetime import datetime
from db.session import SessionLocal, engine
from db.models import Base, Experiment, Assignment, Event, ExperimentStatus
from db.repositories import event_repo, experiment_repo, assignment_repo


def test_get_segment_metrics(db_session):
    # 1. Create Experiment
    exp_id = uuid.uuid4()
    exp = Experiment(
        id=exp_id,
        name=f"Metric_Test_{exp_id.hex[:6]}",
        status=ExperimentStatus.running,
        config={"feature_schema": {}},
        seed=123
    )
    db_session.add(exp)
    db_session.commit()

    # 2. Create Assignments (2 segments, 2 variants each)
    # Segment 1, Control: 10 users, 2 conversions
    for i in range(10):
        user_id = f"user_s1_c_{i}"
        assignment = Assignment(experiment_id=exp_id, user_id=user_id, variant="control", segment_id=1)
        db_session.add(assignment)
        if i < 2:
            event = Event(experiment_id=exp_id, user_id=user_id, event_type="conversion", metric_value=1.0)
            db_session.add(event)

    # Segment 1, Treatment: 10 users, 5 conversions
    for i in range(10):
        user_id = f"user_s1_t_{i}"
        assignment = Assignment(experiment_id=exp_id, user_id=user_id, variant="treatment", segment_id=1)
        db_session.add(assignment)
        if i < 5:
            event = Event(experiment_id=exp_id, user_id=user_id, event_type="conversion", metric_value=1.0)
            db_session.add(event)
    # Segment 2, Control: 5 users (ONLY control and small - should be excluded)
    for i in range(5):
        user_id = f"user_s2_c_{i}"
        assignment = Assignment(experiment_id=exp_id, user_id=user_id, variant="control", segment_id=2)
        db_session.add(assignment)

    # Segment 3, Control: 29 users (Small segment - should be excluded)
    for i in range(29):
        user_id = f"user_s3_c_{i}"
        assignment = Assignment(experiment_id=exp_id, user_id=user_id, variant="control", segment_id=3)
        db_session.add(assignment)
    
    # Segment 3, Treatment: 30 users
    for i in range(30):
        user_id = f"user_s3_t_{i}"
        assignment = Assignment(experiment_id=exp_id, user_id=user_id, variant="treatment", segment_id=3)
        db_session.add(assignment)

    # Segment 4, Control: 30 users, Treatment: 30 users (Balanced and large enough)
    for i in range(30):
        db_session.add(Assignment(experiment_id=exp_id, user_id=f"user_s4_c_{i}", variant="control", segment_id=4))
        db_session.add(Assignment(experiment_id=exp_id, user_id=f"user_s4_t_{i}", variant="treatment", segment_id=4))

    db_session.commit()

    # 3. Call get_segment_metrics
    metrics = event_repo.get_segment_metrics(db_session, exp_id)

    # 4. Assertions
    # Segment 1: Balanced (10 users each) -> EXCLUDED because < 30
    # Segment 2: Only control (5 users) -> EXCLUDED because < 30
    # Segment 3: control=29, treatment=30 -> EXCLUDED because < 30
    # Segment 4: control=30, treatment=30 -> INCLUDED
    
    # Verify Segment 4 is the only one included (2 variants = 2 rows)
    assert len(metrics) == 2
    assert all(m.segment_id == 4 for m in metrics)
    assert all(m.is_testable for m in metrics)
