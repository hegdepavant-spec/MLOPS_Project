"""
Compatibility wrapper for database.py
"""
from src.database import load_database

def load_data():
    res = load_database()
    return res.get("status") == "SUCCESS"

if __name__ == "__main__":
    load_database()
