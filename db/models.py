import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, Enum, ForeignKey, UniqueConstraint, LargeBinary, JSON, Uuid, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, validates

class Base(DeclarativeBase):
    pass

class ExperimentStatus(enum.Enum):
    draft = "draft"
    running = "running"
    stopped = "stopped"
    complete = "complete"

class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    status: Mapped[ExperimentStatus] = mapped_column(Enum(ExperimentStatus), nullable=False, default=ExperimentStatus.draft)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)  # stores MDE, alpha, feature_schema, decision_threshold
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)

class Assignment(Base):
    __tablename__ = "assignments"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("experiments.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    variant: Mapped[str] = mapped_column(String, nullable=False)  # control/treatment
    segment_id: Mapped[int] = mapped_column(Integer, nullable=True)
    enrolled_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("experiment_id", "user_id", name="uq_experiment_user"),
    )

    @validates('segment_id')
    def validate_segment_id(self, key, value):
        """
        Ensures that segment_id is frozen once assigned.
        Raises ValueError if an attempt is made to update a non-null segment_id.
        """
        if self.segment_id is not None and self.segment_id != value:
            raise ValueError(f"segment_id is frozen for user {self.user_id} in experiment {self.experiment_id}.")
        return value

class Event(Base):
    __tablename__ = "events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("experiments.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_event_experiment_user", "experiment_id", "user_id"),
    )

class SegmentModel(Base):
    __tablename__ = "segment_models"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("experiments.id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    scaler_artifact: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    kmeans_artifact: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    n_clusters: Mapped[int] = mapped_column(Integer, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PosteriorSnapshot(Base):
    """
    Append-only log of Bayesian posterior states for each experiment variant.
    Used for auditing, monitoring, and plotting the evolution of conversion rates.
    """
    __tablename__ = "posterior_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("experiments.id"), nullable=False)
    variant: Mapped[str] = mapped_column(String, nullable=False)
    alpha_post: Mapped[float] = mapped_column(Float, nullable=False)
    beta_post: Mapped[float] = mapped_column(Float, nullable=False)
    n_trials: Mapped[int] = mapped_column(Integer, nullable=False)
    n_conversions: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_processed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
