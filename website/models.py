from sqlalchemy.orm import Session
import pandas as pd

from sqlalchemy import Column, Integer, String, create_engine, text, select
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector


# Define connection parameters
params = {
    "host": "localhost",
    "port": 5432,
    "database": "celebsdb",
    "user": "docker",
    "password": "docker",
    "table": "celeb_embeddings",
    "server": "proper_door_database_1"
}

# Define the connection string
# Format: dialect+driver://username:password@host:port/database
connection_string = 'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'.format(**params)

# Create an engine
engine = create_engine(connection_string)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

class CelebEmbedding(Base):
    __tablename__ = 'celeb_embeddings'

    id = Column(Integer, primary_key=True)
    filename = Column(String)
    filepath = Column(String)
    celebname = Column(String)
    target = Column(String)
    embedding = Column(Vector(512))  

# Create the table
Base.metadata.create_all(engine)