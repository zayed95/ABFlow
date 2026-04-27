from celery import shared_task
from sqlalchemy.orm import Session
from sqlalchemy import func
from db.session import SessionLocal
from db.models import Experiment, Assignment, Event, ExperimentStatus
from db.repositories import experiment_repo, posterior_repo
from core.sequential.bayesian import BetaBinomialPosterior
from core.sequential.decision import evaluate_decision, Decision
from core.sequential.frequentist import OBrienFlemingBoundary
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@shared_task
def update_posteriors_task(experiment_id: str):
    """
    Background task to update Bayesian posteriors for an experiment.
    
    1. Fetches new trials (assignments) and conversions (events) since the last snapshot.
    2. Updates the Beta-Binomial posterior for control and treatment.
    3. Evaluates the sequential stopping rule.
    4. Persists new snapshots.
    5. Stops the experiment if a decision is reached.
    """
    db = SessionLocal()
    try:
        exp_uuid = uuid.UUID(experiment_id)
        experiment = experiment_repo.get_experiment(db, exp_uuid)
        
        if not experiment or experiment.status != ExperimentStatus.running:
            logger.info(f"Experiment {experiment_id} not found or not running. Skipping.")
            return

        # Get latest snapshots to resume from
        control_snap = posterior_repo.get_latest_snapshot(db, exp_uuid, "control")
        treatment_snap = posterior_repo.get_latest_snapshot(db, exp_uuid, "treatment")

        # Determine the cutoff for "new" data
        # We look for events that happened AFTER the most recent watermark
        last_processed_at = None
        if control_snap and treatment_snap:
            last_processed_at = max(control_snap.last_processed_at, treatment_snap.last_processed_at)
        
        if not last_processed_at:
            # If no snapshots exist yet, start from when the experiment was created
            last_processed_at = experiment.created_at

        # 1. Query for new trials (Assignments) per variant
        # We also need the max timestamp to update the watermark
        new_trials_query = db.query(
            Assignment.variant, 
            func.count(Assignment.id).label("count"),
            func.max(Assignment.enrolled_at).label("max_enrolled")
        ).filter(
            Assignment.experiment_id == exp_uuid,
            Assignment.enrolled_at > last_processed_at
        ).group_by(Assignment.variant).all()
        
        new_trials = {r.variant: r.count for r in new_trials_query}
        max_enrolled = max([r.max_enrolled for r in new_trials_query if r.max_enrolled] + [last_processed_at])

        # 2. Query for new conversions (Events) per variant
        new_conversions_query = db.query(
            Assignment.variant,
            func.count(Event.id).label("count"),
            func.max(Event.occurred_at).label("max_occurred")
        ).join(
            Event, (Assignment.user_id == Event.user_id) & (Assignment.experiment_id == Event.experiment_id)
        ).filter(
            Event.experiment_id == exp_uuid,
            Event.occurred_at > last_processed_at,
            Event.event_type == 'conversion'
        ).group_by(Assignment.variant).all()
        
        new_conversions = {r.variant: r.count for r in new_conversions_query}
        max_occurred = max([r.max_occurred for r in new_conversions_query if r.max_occurred] + [last_processed_at])

        # The new watermark is the latest data we've seen in this run
        next_watermark = max(max_enrolled, max_occurred)

        # 3. Setup and Update Posteriors
        config = experiment.config
        alpha_prior = config.get("alpha_prior", 1.0)
        beta_prior = config.get("beta_prior", 1.0)

        # Helper to hydrate posterior from DB snapshot or prior
        def hydrate_posterior(snap):
            p = BetaBinomialPosterior(alpha_prior=alpha_prior, beta_prior=beta_prior)
            if snap:
                p.alpha_posterior = snap.alpha_post
                p.beta_posterior = snap.beta_post
            return p

        post_a = hydrate_posterior(control_snap)
        post_b = hydrate_posterior(treatment_snap)

        # Perform the actual update with the delta data
        post_a.update(
            n_new_conversions=new_conversions.get("control", 0),
            n_new_trials=new_trials.get("control", 0)
        )
        post_b.update(
            n_new_conversions=new_conversions.get("treatment", 0),
            n_new_trials=new_trials.get("treatment", 0)
        )

        # 4. Evaluate Decision based on Mode
        mode = config.get("mode", "bayesian")
        
        if mode == "frequentist":
            # Frequentist Look: Each task run is considered a sequential look
            # Calculate current look number based on existing snapshots
            look_number = db.query(PosteriorSnapshot).filter(
                PosteriorSnapshot.experiment_id == exp_uuid,
                PosteriorSnapshot.variant == "control"
            ).count() + 1
            
            n_planned_looks = config.get("n_planned_looks", 5)
            alpha = config.get("alpha", 0.05)
            
            obf = OBrienFlemingBoundary(alpha=alpha, n_planned_looks=n_planned_looks)
            
            # Extract cumulative counts
            n_a = int(post_a.alpha_posterior - post_a.alpha_prior + post_a.beta_posterior - post_a.beta_prior)
            c_a = int(post_a.alpha_posterior - post_a.alpha_prior)
            n_b = int(post_b.alpha_posterior - post_b.alpha_prior + post_b.beta_posterior - post_b.beta_prior)
            c_b = int(post_b.alpha_posterior - post_b.alpha_prior)
            
            # We cap the look number at n_planned_looks
            current_look = min(look_number, n_planned_looks)
            test_res = obf.test_at_look(n_a, c_a, n_b, c_b, current_look)
            
            if test_res["reject"]:
                # If B's rate is higher, it's a winner
                if (c_b / n_b if n_b > 0 else 0) > (c_a / n_a if n_a > 0 else 0):
                    decision = Decision.STOP_WINNER
                else:
                    decision = Decision.STOP_NULL
            else:
                decision = Decision.CONTINUE
        else:
            # Bayesian (Default)
            result = evaluate_decision(
                post_a, post_b, 
                threshold_win=config.get("threshold_win", 0.95),
                threshold_null=config.get("threshold_null", 0.05),
                min_samples=config.get("min_samples", 100)
            )
            decision = result.decision

        # 5. Persist Snapshots
        def persist_snapshot(variant, post):
            total_conv = int(post.alpha_posterior - post.alpha_prior)
            total_trials = int((post.alpha_posterior - post.alpha_prior) + (post.beta_posterior - post.beta_prior))
            
            posterior_repo.save_snapshot(
                db, exp_uuid, variant,
                alpha_post=post.alpha_posterior,
                beta_post=post.beta_posterior,
                n_trials=total_trials,
                n_conversions=total_conv,
                last_processed_at=next_watermark
            )

        persist_snapshot("control", post_a)
        persist_snapshot("treatment", post_b)

        # 6. Stop experiment if decision reached
        if decision in [Decision.STOP_WINNER, Decision.STOP_NULL]:
            logger.info(f"Stopping experiment {experiment_id} due to decision: {decision}")
            experiment_repo.update_status(db, exp_uuid, ExperimentStatus.stopped)

    except Exception as e:
        logger.error(f"Error in update_posteriors_task for {experiment_id}: {str(e)}")
        raise
    finally:
        db.close()


@shared_task
def batch_update_all_posteriors_task():
    """
    Periodic task to find all running experiments and trigger an update for each.
    """
    db = SessionLocal()
    try:
        running_experiments = db.query(Experiment).filter(
            Experiment.status == ExperimentStatus.running
        ).all()
        
        for exp in running_experiments:
            update_posteriors_task.delay(str(exp.id))
            
    finally:
        db.close()
