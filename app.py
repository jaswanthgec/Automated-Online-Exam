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

# The master teacher password
MASTER_TEACHER_PASSWORD = "teacher_secret"

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

# Placeholder pages for students
def student_quiz():
    st.write("Student Quiz Page")

def student_sample_quiz():
    st.write("Student Sample Quiz Page")

def student_previous_scores():
    st.write("Student Previous Scores Page")

def student_profile():
    st.write("Student Profile Page")

# Placeholder pages for teachers
def teacher_create_quiz():
    st.write("Teacher Create Quiz Page")

def teacher_create_team():
    st.write("Teacher Create Team Page")

def teacher_results():
    st.write("Teacher Results Page")

def teacher_exam_history():
    st.write("Teacher Exam History Page")

def teacher_sample_quiz():
    st.write("Teacher Sample Quiz Page")

def teacher_profile():
    st.write("Teacher Profile Page")

def teacher_help():
    st.write("Teacher Help Page")

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
        role = st.selectbox("Role", ["student", "teacher"], key="signup_role")
        if role == "teacher":
            teacher_password = st.text_input("Teacher Password", type="password", key="teacher_password")
        else:
            teacher_password = None

        if st.button("Signup"):
            if role == "teacher" and teacher_password != MASTER_TEACHER_PASSWORD:
                st.error("Invalid teacher password")
            else:
                if add_user(new_username, new_password, role):
                    st.success(f"User {new_username} registered successfully as {role}")
                else:
                    st.error("Username already taken")

    # Display navigation based on role
    if st.session_state.logged_in:
        role = st.session_state.role
        if role == "student":
            student_nav()
        elif role == "teacher":
            teacher_nav()

def student_nav():
    # Student-specific navigation
    pages = {
        "Quiz with Code": student_quiz,
        "Sample Quiz": student_sample_quiz,
        "Previous Exam Scores": student_previous_scores,
        "Profile": student_profile
    }
    page = st.sidebar.selectbox("Select a page", list(pages.keys()))
    pages[page]()

def teacher_nav():
    # Teacher-specific navigation
    pages = {
        "Create Quiz": teacher_create_quiz,
        "Create Team": teacher_create_team,
        "Results": teacher_results,
        "Previous Exams History": teacher_exam_history,
        "Sample Quiz": teacher_sample_quiz,
        "Profile": teacher_profile,
        "Help": teacher_help
    }
    page = st.sidebar.selectbox("Select a page", list(pages.keys()))
    pages[page]()

if __name__ == "__main__":
    main()
