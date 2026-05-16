import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import db.session

# Use in-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# PATCH the global SessionLocal and engine BEFORE any other imports that might use them
db.session.SessionLocal = TestingSessionLocal
db.session.engine = engine

# Now import the rest
from db.models import Base
from api.main import app

@pytest.fixture(scope="session", autouse=True)
def create_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(autouse=True)
def override_get_db(db_session):
    def _get_db_override():
        yield db_session
    
    app.dependency_overrides[db.session.get_db] = _get_db_override
    yield
    app.dependency_overrides.clear()
