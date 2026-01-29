import sqlite3
import pandas as pd
import os

DB_PATH = "youtube.db"

print("🔥 DB LOADER STARTED")

conn = sqlite3.connect(DB_PATH)

clean_df = pd.read_csv("data/processed/clean_youtube_data.csv", encoding="latin1")
featured_df = pd.read_csv("data/processed/featured_youtube_data.csv", encoding="utf-8")

clean_df.to_sql("clean_youtube_data", conn, if_exists="replace", index=False)
featured_df.to_sql("featured_youtube_data", conn, if_exists="replace", index=False)

print("✅ Tables created:")
print("- clean_youtube_data")
print("- featured_youtube_data")

conn.close()
print("🎯 DATABASE READY")