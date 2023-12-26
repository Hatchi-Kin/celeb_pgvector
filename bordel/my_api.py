from fastapi import FastAPI
import uvicorn
from sqlalchemy.orm import Session
import pandas as pd

from sqlalchemy import Column, Integer, String, create_engine, text, select
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector

app = FastAPI()

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


@app.get("/random_embeddings")
def read_random_embeddings():
    random_celeb_embedding = session.query(CelebEmbedding).order_by(func.random()).first()
    query = select(CelebEmbedding).order_by(CelebEmbedding.embedding.l2_distance(random_celeb_embedding.embedding)).limit(5)
    results = session.scalars(query).all()
    top_5 = []
    for result in results:
        top_5.append(result.celebname)
    return top_5


# This is how you should run the FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)