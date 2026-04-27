from celery import Celery
import os

# Celery configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "abflow_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks.update_posteriors"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "update-all-posteriors-every-5-minutes": {
            "task": "tasks.update_posteriors.batch_update_all_posteriors_task",
            "schedule": 300.0,  # 5 minutes in seconds
        },
    },
)
