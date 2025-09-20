import os, duckdb, pandas as pd
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv("DUCKDB_PATH", "fw.duckdb")

df = pd.read_csv("data/prices/sample_prices.csv", parse_dates=["month"])
con = duckdb.connect(DB_PATH)
con.execute("""CREATE TABLE IF NOT EXISTS prices(
  admin2 TEXT, commodity TEXT, month DATE, price DOUBLE, unit TEXT
);
""")
con.execute("""INSERT INTO prices SELECT * FROM df""")
print(f"Loaded {len(df)} rows into {DB_PATH}::prices")
con.close()
