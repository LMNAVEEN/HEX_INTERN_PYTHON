# Run: python 03_tools_and_agents/sqlite_react_agent.py
# Tech: LangGraph prebuilt ReAct agent + LangChain Tool + SQLite + Groq (LLaMA)
# Purpose: ReAct agent that answers natural language questions by querying a SQLite employee database

import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.tools import tool
from langchain_groq import ChatGroq
# Import the LangGraph Prebuilt React Agent architecture
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

DB_FILE_PATH = os.path.join(BASE_DIR, "data", "databases", "test.db")

def init_mock_database():
    """Creates a temporary database populated with employee records."""
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    
    # Create sample table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL
        )
    """)

    # Populate mock data
    cursor.execute("SELECT COUNT(*) FROM employees")
    row_count = cursor.fetchone()[0]
    
    if row_count == 0:
        mock_data = [
            ("Alice", "Senior Android Developer", "Engineering", 115000),
            ("Bob", "Backend Engineer", "Engineering", 98000),
            ("Charlie", "UI/UX Designer", "Design", 85000),
            ("Diana", "Product Manager", "Product", 120000)
        ]
        cursor.executemany(
            "INSERT INTO employees (name, role, department, salary) VALUES (?, ?, ?, ?)", 
            mock_data
        )
        conn.commit()
        print(f"-> Database freshly initialized and seeded at: {DB_FILE_PATH}")
    else:
        print(f"-> Existing database found with {row_count} records. Skipping data insertion.")
        
    conn.close()

# Instantiate our local test database instance
init_mock_database()


# ==========================================
# 2. LANGCHAIN TOOL DEFINITION
# ==========================================
@tool
def query_sqlite_database(query: str) -> str:
    """
    Executes a read-only SQL query against the local SQLite database 
    and returns the structured text results. Use this tool whenever 
    asked for details regarding staff, departments, or salaries.
    """
    try:
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            return "Query completed successfully, but returned no rows."
            
        # Format the output matching the schema column values cleanly
        return "\n".join([str(row) for row in rows])
    except sqlite3.Error as e:
        return f"Database Error: {str(e)}"


def main():
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.3
    )

    # Collect the available tools matching W Shamim's design pattern
    tools = [query_sqlite_database]
    
    # Create the reactive loop agent structure
    agent_executor = create_react_agent(llm, tools)

    user_query = "Can you look into the database and find the average salary of the Engineering department?"
    print(f"User Request: {user_query}\n")

    response = agent_executor.invoke({"messages": [("user", user_query)]})
    
    # Extract and view the final response message text content
    print("Agent Final Output:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()    
    