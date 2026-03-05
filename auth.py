from database import c, conn
import hashlib

# Hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Register user
def register_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

# Login user
def login_user(username, password):
    c.execute("SELECT id FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    result = c.fetchone()
    if result:
        return result[0]  # return user_id
    return None