import sqlite3
import streamlit as st

# Connect to SQLite database (or create it)
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT,
        role TEXT
    )
''')
conn.commit()

# The master admin password
MASTER_ADMIN_PASSWORD = "admin_secret"

# Function to verify credentials
def verify_credentials(username, password):
    c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    if result:
        return result[0]
    return None

# Function to add a new user
def add_user(username, password, role):
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# Main function to run the app
def main():
    # Set the title of the app
    st.title("Login and Signup Page")

    # Create a session state for login status and role
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'role' not in st.session_state:
        st.session_state.role = None

    # Tabs for Login and Signup
    tabs = st.tabs(["Login", "Signup"])

    # Login tab
    with tabs[0]:
        if not st.session_state.logged_in:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                role = verify_credentials(username, password)
                if role:
                    st.session_state.logged_in = True
                    st.session_state.role = role
                    st.success(f"Login successful as {role}")
                else:
                    st.error("Invalid username or password")
        else:
            st.write(f"Welcome, you are logged in as {st.session_state.role}!")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.role = None

    # Signup tab
    with tabs[1]:
        new_username = st.text_input("New Username", key="signup_username")
        new_password = st.text_input("New Password", type="password", key="signup_password")
        role = st.selectbox("Role", ["user", "admin"], key="signup_role")
        if role == "admin":
            admin_password = st.text_input("Admin Password", type="password", key="admin_password")
        else:
            admin_password = None

        if st.button("Signup"):
            if role == "admin" and admin_password != MASTER_ADMIN_PASSWORD:
                st.error("Invalid admin password")
            else:
                if add_user(new_username, new_password, role):
                    st.success(f"User {new_username} registered successfully as {role}")
                else:
                    st.error("Username already taken")

if __name__ == "__main__":
    main()
