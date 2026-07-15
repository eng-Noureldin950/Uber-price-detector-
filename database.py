import sqlite3
from datetime import datetime

class PredictionLogger:
    def __init__(self, db_path="predictions.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pickup_lat REAL,
                pickup_lon REAL,
                dropoff_lat REAL,
                dropoff_lon REAL,
                distance_km REAL,
                predicted_fare REAL,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_request(self, p_lat, p_lon, d_lat, d_lon, distance, fare):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO requests (pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, distance_km, predicted_fare, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (p_lat, p_lon, d_lat, d_lon, distance, float(fare), datetime.now().isoformat()))
        conn.commit()
        conn.close()
