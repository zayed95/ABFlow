import pytest
import uuid
import random
from datetime import datetime, timedelta
from db.models import Experiment, Assignment, Event, ExperimentStatus
from core.uplift.data import prepare_dataset

def test_prepare_dataset_dimensions(db_session):
    """
    Test that prepare_dataset returns correct dimension matrices for a fixture
    of 100 control and 100 treatment users.
    """
    exp_id = uuid.uuid4()
    exp = Experiment(
        id=exp_id,
        name=f"Test_Uplift_{exp_id}",
        status=ExperimentStatus.complete,
        config={"outcome_window_hours": 72},
        seed=42
    )
    db_session.add(exp)
    
    enrolled_at = datetime.utcnow() - timedelta(days=5)
    
    # 100 control, 100 treatment
    for i in range(100):
        # control
        assignment_c = Assignment(
            experiment_id=exp_id,
            user_id=f"c_user_{i}",
            variant="control",
            enrolled_at=enrolled_at,
            features={"recency": random.random(), "frequency": random.randint(1, 10)}
        )
        # treatment
        assignment_t = Assignment(
            experiment_id=exp_id,
            user_id=f"t_user_{i}",
            variant="treatment",
            enrolled_at=enrolled_at,
            features={"recency": random.random(), "frequency": random.randint(1, 10)}
        )
        db_session.add_all([assignment_c, assignment_t])
        
        # Add some events within window
        event_c = Event(
            experiment_id=exp_id,
            user_id=f"c_user_{i}",
            event_type="view",
            metric_value=1.0,
            occurred_at=enrolled_at + timedelta(hours=10)
        )
        event_t = Event(
            experiment_id=exp_id,
            user_id=f"t_user_{i}",
            event_type="conversion",
            metric_value=5.0,
            occurred_at=enrolled_at + timedelta(hours=20)
        )
        # Add an event outside window (should be ignored)
        event_out = Event(
            experiment_id=exp_id,
            user_id=f"c_user_{i}",
            event_type="conversion",
            metric_value=10.0,
            occurred_at=enrolled_at + timedelta(hours=80)
        )
        db_session.add_all([event_c, event_t, event_out])
        
    db_session.commit()
    
    # Run prepare_dataset
    dataset = prepare_dataset(db_session, str(exp_id))
    
    # Assertions
    assert dataset.X_control.shape[0] == 100
    assert dataset.X_treatment.shape[0] == 100
    assert dataset.y_control.shape[0] == 100
    assert dataset.y_treatment.shape[0] == 100
    
    num_features = len(dataset.feature_names)
    assert dataset.X_control.shape[1] == num_features
    assert dataset.X_treatment.shape[1] == num_features

    # Verify out-of-window event was ignored (c_user y should be 1.0, not 11.0)
    # y_control is a numpy array. The first element corresponds to whichever control user is first in the list
    assert all(y == 1.0 for y in dataset.y_control)
    assert all(y == 5.0 for y in dataset.y_treatment)
