import pandas as pd
from sqlalchemy import create_engine, URL
from getpass import getpass

password = getpass("Enter PostgreSQL password: ")

url = URL.create(
    "postgresql+psycopg2",
    username="postgres",
    password=password,
    host="localhost",
    port=5432,
    database="retainai"
)

engine = create_engine(url)

df = pd.read_csv("data/raw/telco_churn.csv")

df.to_sql(
    "customers",
    engine,
    if_exists="fail",
    index=False
)

print("Dataset uploaded successfully!")
print("Total customers:", len(df))