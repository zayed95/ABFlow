import logging
import uuid
from dataclasses import dataclass
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func, case, distinct
from db.models import Assignment, Event

logger = logging.getLogger(__name__)

@dataclass
class SegmentMetric:
    """
    Represents aggregated metrics for a specific segment and variant combination.
    """
    segment_id: Optional[int]
    variant: str
    n_users: int
    n_conversions: int
    conversion_rate: float
    is_testable: bool

def get_segment_metrics(db: Session, experiment_id: uuid.UUID) -> list[SegmentMetric]:
    """
    Returns a list of SegmentMetric objects for a given experiment.
    Identifies imbalanced segments as untestable.
    Excludes segments with fewer than 30 users in either variant.

    :param db: SQLAlchemy session
    :param experiment_id: UUID of the experiment
    :return: List of SegmentMetric objects
    """
    # Aggregate conversions per user first to avoid row multiplication in the join
    user_conversions = (
        db.query(
            Event.user_id,
            func.sum(case((Event.event_type == 'conversion', 1), else_=0)).label('conversions')
        )
        .filter(Event.experiment_id == experiment_id)
        .group_by(Event.user_id)
        .subquery()
    )

    # Join assignments with aggregated user conversions
    results = (
        db.query(
            Assignment.segment_id,
            Assignment.variant,
            func.count(Assignment.user_id).label('n_users'),
            func.sum(func.coalesce(user_conversions.c.conversions, 0)).label('n_conversions')
        )
        .outerjoin(
            user_conversions,
            Assignment.user_id == user_conversions.c.user_id
        )
        .filter(Assignment.experiment_id == experiment_id)
        .group_by(Assignment.segment_id, Assignment.variant)
        .all()
    )

    # 1. Collect statistics per segment
    segment_stats = {}
    for segment_id, variant, n_users, _ in results:
        if segment_id not in segment_stats:
            segment_stats[segment_id] = {"control": 0, "treatment": 0}
        segment_stats[segment_id][variant] = n_users

    # 2. Filter segments and log warnings for those excluded
    testable_segments = {} # segment_id -> is_testable
    excluded_segments = set()
    
    MIN_USERS = 30

    for segment_id, stats in segment_stats.items():
        control_n = stats.get("control", 0)
        treatment_n = stats.get("treatment", 0)
        
        is_balanced = "control" in stats and "treatment" in stats
        is_large_enough = control_n >= MIN_USERS and treatment_n >= MIN_USERS
        
        if not is_large_enough:
            logger.warning(
                f"Segment {segment_id} excluded from HTE analysis: "
                f"insufficient sample size (control={control_n}, treatment={treatment_n})"
            )
            excluded_segments.add(segment_id)
        else:
            testable_segments[segment_id] = is_balanced

    # 3. Build metrics list, skipping excluded segments
    metrics = []
    for segment_id, variant, n_users, n_conversions in results:
        if segment_id in excluded_segments:
            continue
            
        n_conversions = n_conversions or 0
        conversion_rate = n_conversions / n_users if n_users > 0 else 0.0
        
        metrics.append(SegmentMetric(
            segment_id=segment_id,
            variant=variant,
            n_users=n_users,
            n_conversions=n_conversions,
            conversion_rate=conversion_rate,
            is_testable=testable_segments.get(segment_id, False)
        ))
    
    return metrics
