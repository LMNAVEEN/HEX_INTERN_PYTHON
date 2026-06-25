# Run: python 03_tools_and_agents/resume_pipeline_groq.py
# Tech: LangGraph + Groq SDK (raw) + pdfplumber
# Purpose: Automated resume screening pipeline — PDF parsing, scoring, interview simulation, and hiring decision

import os
import glob
import json
from typing import TypedDict, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pdfplumber
from groq import Groq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

JD_FOLDER = os.path.join(BASE_DIR, "jd")
RESUME_FOLDER = os.path.join(BASE_DIR, "resumes")
HIRE_THRESHOLD = 85

PANEL_SLOTS = ["Mon 10AM", "Mon 2PM", "Tue 10AM", "Tue 2PM", "Wed 10AM"]
booked_slots: set[str] = set()


def load_job_description() -> str:
    """Extract JD text from whichever file is sitting in jd/ (pdf or txt)."""
    jd_files = glob.glob(os.path.join(JD_FOLDER, "*"))
    if not jd_files:
        raise FileNotFoundError(f"No JD file found in '{JD_FOLDER}/'")
    jd_path = jd_files[0]
    if jd_path.lower().endswith(".pdf"):
        with pdfplumber.open(jd_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    with open(jd_path, "r", encoding="utf-8") as f:
        return f.read()


JOB_DESCRIPTION = load_job_description()


class CandidateState(TypedDict):
    file_name: str
    resume_text: str
    candidate_name: Optional[str]
    screening_score: Optional[int]
    screening_status: Optional[str]
    screening_reason: Optional[str]
    interview_slot: Optional[str]
    simulated_response: Optional[str]
    evaluation_result: Optional[dict]
    status: Optional[str]


def call_llm(prompt: str, temperature: float = 0.4) -> str:
    """Single LLM call - no retries, keeps token usage predictable."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def extract_json(text: str) -> dict:
    """Pull the JSON object out of an LLM response."""
    start, end = text.find("{"), text.rfind("}") + 1
    return json.loads(text[start:end])


# ---------------- NODE 0: PDF -> text ----------------
def convert_to_llm_input(state: CandidateState) -> CandidateState:
    """Extract raw resume text from the PDF on disk."""
    try:
        with pdfplumber.open(state["file_name"]) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        state["resume_text"] = text or "[No text could be extracted from this PDF]"
    except Exception as e:
        state["resume_text"] = f"[Error reading PDF: {e}]"
    return state


# ---------------- NODE 1: Resume Screening ----------------
def screen_resume(state: CandidateState) -> CandidateState:
    """Score the resume holistically against the actual job description."""
    prompt = f"""
You are screening a resume against the following job description.

Job Description:
{JOB_DESCRIPTION}

Resume:
{state['resume_text']}

Extract the candidate's full name from the resume text.

Evaluate the candidate HOLISTICALLY against everything in the job
description above - required skills, responsibilities, qualifications, and
experience expectations. Do not judge years-of-experience or job title in
isolation; weigh actual project/work evidence that demonstrates the
required skills, not just keyword presence on the resume.

Combine all relevant factors into one overall score (0-100).

Respond ONLY in JSON:
{{
  "candidate_name": "<full name>",
  "score": <int 0-100>,
  "status": "selected" or "rejected",
  "reason": "<2-3 sentences citing specific factors from the JD that drove this decision>"
}}
"""
    try:
        result = extract_json(call_llm(prompt))
        state["candidate_name"] = result.get("candidate_name", "Unknown")
        state["screening_score"] = result["score"]
        state["screening_status"] = result["status"]
        state["screening_reason"] = result["reason"]
    except Exception as e:
        state["candidate_name"] = "Unknown"
        state["screening_score"] = 0
        state["screening_status"] = "rejected"
        state["screening_reason"] = f"Could not screen resume: {e}"
    return state


def route_after_screening(state: CandidateState) -> str:
    """Conditional edge: selected candidates move on, others end early."""
    return "schedule_interview" if state["screening_status"] == "selected" else "rejected_end"


# ---------------- NODE 2: Interview Scheduling ----------------
def schedule_interview(state: CandidateState) -> CandidateState:
    """Assign the next free panel slot, avoiding double-booking."""
    for slot in PANEL_SLOTS:
        if slot not in booked_slots:
            state["interview_slot"] = slot
            booked_slots.add(slot)  # FIX: actually reserve the slot so the next candidate doesn't get it too
            break
    else:
        state["interview_slot"] = "No slots available"
    return state


# ---------------- NODE 3: Simulate candidate response ----------------
def simulate_response(state: CandidateState) -> CandidateState:
    """Generate a realistic, resume-grounded interview answer."""
    prompt = f"""
Simulate how this candidate would realistically answer in an interview for
this role:

Job Description:
{JOB_DESCRIPTION}

Resume:
{state['resume_text']}

Question: "Describe a project where you applied embeddings or RAG, and a
challenge you faced while building it."

Stay honest to their apparent skill level based on the resume - do not make
them sound more or less expert than the resume supports. Give a realistic
4-5 sentence answer in the candidate's voice.
"""
    try:
        # FIX: write to 'simulated_response' to match the CandidateState TypedDict
        state["simulated_response"] = call_llm(prompt, temperature=0.6)
    except Exception as e:
        state["simulated_response"] = f"[Error generating response: {e}]"
    return state


# ---------------- NODE 4: Evaluate candidate ----------------
def evaluate_candidate(state: CandidateState) -> CandidateState:
    """Score the simulated answer and apply the hiring bar."""
    prompt = f"""
Evaluate this simulated interview answer against the following job
description:

Job Description:
{JOB_DESCRIPTION}

Candidate: {state['candidate_name']}
Answer:
{state['simulated_response']}

Score 0-100 based on technical depth, clarity, and relevance to the
responsibilities and skills in the job description above.
A candidate is only "hired" if interview_score is STRICTLY GREATER THAN
{HIRE_THRESHOLD} - otherwise "not_hired".

Respond ONLY in JSON:
{{
  "interview_score": <int 0-100>,
  "verdict": "hired" or "not_hired",
  "performance_summary": "<2-3 sentences on how they performed>",
  "areas_for_improvement": "<2-3 sentences on what specifically to improve>"
}}
"""
    try:
        result = extract_json(call_llm(prompt, temperature=0.3))
        state["evaluation_result"] = result
        state["status"] = result["verdict"]
    except Exception as e:
        state["evaluation_result"] = {
            "interview_score": 0,
            "verdict": "not_hired",
            "performance_summary": "N/A",
            "areas_for_improvement": f"Evaluation error: {e}",
        }
        state["status"] = "not_hired"
    return state


def rejected_end(state: CandidateState) -> CandidateState:
    """Terminal node for candidates who did not pass screening."""
    state["status"] = "rejected"
    return state


# ---------------- GRAPH WIRING ----------------
graph = StateGraph(CandidateState)
graph.add_node("convert_to_llm_input", convert_to_llm_input)
graph.add_node("screen_resume", screen_resume)
graph.add_node("schedule_interview", schedule_interview)
graph.add_node("simulate_response", simulate_response)
graph.add_node("evaluate_candidate", evaluate_candidate)
graph.add_node("rejected_end", rejected_end)

graph.add_edge(START, "convert_to_llm_input")
graph.add_edge("convert_to_llm_input", "screen_resume")
graph.add_conditional_edges(
    "screen_resume",
    route_after_screening,
    {"schedule_interview": "schedule_interview", "rejected_end": "rejected_end"},
)
graph.add_edge("schedule_interview", "simulate_response")
graph.add_edge("simulate_response", "evaluate_candidate")
graph.add_edge("evaluate_candidate", END)
graph.add_edge("rejected_end", END)

app = graph.compile()


def print_candidate_report(result: dict, file_name: str) -> None:
    """Clean per-candidate console output."""
    print(f"\n{'=' * 60}")
    print(f"Candidate: {result.get('candidate_name', 'Unknown')} ({os.path.basename(file_name)})")
    print(f"Screening score: {result.get('screening_score')} | Status: {result['status']}")

    if result["screening_status"] == "rejected":
        print(f"Reason not selected: {result.get('screening_reason')}")
        return

    print(f"Why they advanced: {result.get('screening_reason')}")
    print(f"Interview slot: {result.get('interview_slot')}")

    ev = result.get("evaluation_result", {})
    print(f"Interview score: {ev.get('interview_score')}/100")
    print(f"Performance: {ev.get('performance_summary')}")
    print(f"Areas to improve: {ev.get('areas_for_improvement')}")


def print_summary(all_results: list[dict]) -> None:
    """Ranked leaderboard at the end of the run."""
    hired = sorted(
        [r for r in all_results if r["status"] == "hired"],
        key=lambda r: r["evaluation_result"]["interview_score"],
        reverse=True,
    )
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(all_results)} processed | {len(hired)} hired")
    for rank, r in enumerate(hired, start=1):
        score = r["evaluation_result"]["interview_score"]
        print(f"  {rank}. {r['candidate_name']} - {score}/100")


# ---------------- RUN OVER ALL RESUMES ----------------
if __name__ == "__main__":
    resume_files = glob.glob(os.path.join(RESUME_FOLDER, "*.pdf"))
    all_results = []

    for path in resume_files:
        result = app.invoke({"file_name": path})
        all_results.append(result)
        print_candidate_report(result, path)

    print_summary(all_results)