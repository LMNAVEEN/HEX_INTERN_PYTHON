# Run: python 04_crewai/ai_trends_crew.py
# Tech: CrewAI + Google Gemini
# Purpose: Multi-agent crew — Research Analyst + Content Writer — produce an AI trends blog post

import os
os.environ["CREWAI_DISABLE_IPYTHON"] = "true"

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Load environment variables from .env file
load_dotenv()

GEMINI_MODEL = "gemini/gemini-3.1-flash-lite"
# Initialize Gemini 2.5 Flash model using CrewAI's LLM class
gemini_llm = LLM(
    model=GEMINI_MODEL,
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

researcher = Agent(
    role="Research Analyst",
    goal="Research and analyze information on given topics",
    backstory="You are an expert researcher with a keen eye for detail and accuracy. "
              "You excel at finding relevant information and presenting it clearly.",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

researcher = Agent(
    role="Research Analyst",
    goal="Research and analyze information on given topics",
    backstory="You are an expert researcher with a keen eye for detail and accuracy. "
              "You excel at finding relevant information and presenting it clearly.",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging and informative content based on research",
    backstory="You are a skilled writer who can transform research findings into "
              "compelling and easy-to-understand content for various audiences.",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

research_task = Task(
    description="Research the latest trends in artificial intelligence and machine learning. "
                "Focus on developments in 2024 and 2025. Provide key insights and statistics.",
    agent=researcher,
    expected_output="A comprehensive research report with key AI/ML trends, statistics, and insights"
)

writing_task = Task(
    description="Based on the research findings, write a blog post about AI trends. "
                "Make it engaging for a general audience, include examples, and keep it informative but accessible.",
    agent=writer,
    expected_output="A well-structured blog post about AI trends (800-1000 words)",
    context=[research_task]  # This task depends on the research task output
)

ai_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True
)



if __name__ == "__main__":
    print("Starting AI Trend Analysis Crew...")
    result = ai_crew.kickoff()
    print("\n" + "="*50)
    print("FINAL RESULT:")
    print("="*50)
    print(result)







