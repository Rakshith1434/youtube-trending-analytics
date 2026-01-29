import sqlite3
import pandas as pd
import os

print("🔥 DATABASE SCRIPT STARTED")

DB_PATH = "youtube.db"
CSV_PATH = "data/processed/clean_youtube_data.csv"

def create_database():
    print("🔹 Connecting to SQLite...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS youtube_data (
            video_id TEXT,
            title TEXT,
            channel_title TEXT,
            category_id INTEGER,
            publish_time TEXT,
            tags TEXT,
            views INTEGER,
            likes INTEGER,
            dislikes INTEGER,
            comment_count INTEGER,
            region TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database & table created")

def load_csv_to_db():
    print("🔹 Loading CSV...")
    df = pd.read_csv(CSV_PATH, encoding="latin1")

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("youtube_data", conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Inserted {len(df)} rows into database")

if __name__ == "__main__":
    print("🔥 MAIN BLOCK RUNNING")
    create_database()
    load_csv_to_db()
    print("🎯 DATABASE SETUP COMPLETE")