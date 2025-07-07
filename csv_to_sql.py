import pandas as pd
import sqlite3
df = pd.read_csv("your_data.csv")
conn = sqlite3.connect("marketing.db")
df.to_sql("marketing_data", conn, if_exists="replace", index=False)
conn.commit()
conn.close()