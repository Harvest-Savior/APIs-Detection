from google.cloud.sql.connector import Connector
import sqlalchemy

connector = Connector()

def getconn():
    conn = connector.connect(
        "harvestsavior-425512:asia-southeast2:harvestsavior",
        "pymysql",
        user="root",
        password="hv12345",
        db="hs_db",
    )
    return conn

def create_connection_pool():
    return sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        pool_size=5,
        max_overflow=10  
    )