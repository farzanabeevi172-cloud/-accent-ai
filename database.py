import sqlite3

# Connect to local SQLite database
conn = sqlite3.connect("accentai.db", check_same_thread=False)
c = conn.cursor()

# Create users table
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")

# Create history table
c.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    accent TEXT,
    confidence REAL,
    timestamp TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")
conn.commit()