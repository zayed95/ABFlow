from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from config import Settings
from db.session import engine
from db.models import Base
from api.routers.experiments import experiment_router
from api.routers.assignments import assignment_router
from api.routers.results import results_router
from api.routers.uplift import uplift_router

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup
    # Note: In production, we usually use migrations (Alembic)
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(
    title="ABFlow API",
    description="API for ABFlow Experimentation Platform",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(IntegrityError)
async def integrity_exception_handler(request: Request, exc: IntegrityError):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": "An experiment with this name already exists or another integrity constraint was violated."},
    )

# Include routers
app.include_router(experiment_router)
app.include_router(assignment_router)
app.include_router(results_router)
app.include_router(uplift_router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to ABFlow API",
        "docs": "/docs",
        "status": "online"
    }