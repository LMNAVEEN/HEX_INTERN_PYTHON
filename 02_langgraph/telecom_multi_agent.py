# Run: python 02_langgraph/telecom_multi_agent.py
# Tech: LangGraph + Groq (LLaMA) + Multi-Agent (Competitor A/B, Evaluator, Adversary, Refiner, Federated)
# Purpose: Multi-agent telecom customer support system with adversarial refinement

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()


GROQ_MODEL = "llama-3.3-70b-versatile"

llm_competitor = ChatGroq(model=GROQ_MODEL, temperature=0.7)
llm_strategic  = ChatGroq(model=GROQ_MODEL, temperature=0.2)
llm_evaluator  = ChatGroq(model=GROQ_MODEL, temperature=0.0)
llm_adversary  = ChatGroq(model=GROQ_MODEL, temperature=0.9)
llm_refiner    = ChatGroq(model=GROQ_MODEL, temperature=0.0)
llm_federated  = ChatGroq(model=GROQ_MODEL, temperature=0.0)

TELECOM_POLICIES = """
1. International roaming charges apply when customer device connects to foreign network.
2. Billing disputes must be resolved within 7 days.
3. Compensation up to 25% allowed for first-time billing errors.
4. Detailed itemized bill must be provided upon request.
"""

class SupportState(TypedDict):
    messages: Annotated[List, add_messages]
    solution_a: str
    solution_b: str
    selected_solution: str
    adversarial_feedback: str
    refined_solution: str
    federated_solution: str


def competitor_a(state: SupportState):
    print("[Agent A] generating solution...")
    response = llm_competitor.invoke(
        f"Customer query: {state['messages'][-1].content}\n"
        "Provide cost-efficient company resolution."
    )
    return {"solution_a": response.content}

def competitor_b(state: SupportState):
    print("[Agent B] generating solution...")
    response = llm_strategic.invoke(
        f"Customer query: {state['messages'][-1].content}\n"
        "Provide highly empathetic customer satisfaction resolution."
    )
    return {"solution_b": response.content}
    
def evaluator(state: SupportState):
    print("[Evaluator] evaluating competing solutions...")
    response = llm_evaluator.invoke(
        f"""
Customer Query:
{state['messages'][-1].content}

Solution A:
{state['solution_a']}

Solution B:
{state['solution_b']}

Select the better resolution and explain briefly.
Return the selected full solution only.
"""
    )
    return {"selected_solution": response.content}

def adversarial_agent(state: SupportState):
    print("[Adversary] attacking solution...")
    response = llm_adversary.invoke(
        f"""
You are an unhappy telecom customer.

Given this support response:
{state['selected_solution']}

Find weaknesses, missing compensation,
and possible dissatisfaction reasons.
"""
    )
    return {"adversarial_feedback": response.content}
    
def refiner_agent(state: SupportState):
    print("[Refiner] improving solution...")
    response = llm_refiner.invoke(
        f"""
Original Solution:
{state['selected_solution']}

Customer Complaints:
{state['adversarial_feedback']}

Telecom Policies:
{TELECOM_POLICIES}

Improve solution addressing ALL complaints.
"""
    )
    return {"refined_solution": response.content}
    
def federated_agent(state: SupportState):
    print("[Federated] consolidating insights...")
    response = llm_federated.invoke(
        f"""
Combine insights from:
- Competitive solutions
- Adversarial weaknesses
- Refined corrections
- Telecom policy

Produce final enterprise-grade telecom resolution.

Solution A:
{state['solution_a']}

Solution B:
{state['solution_b']}

Adversarial Feedback:
{state['adversarial_feedback']}

Refined Solution:
{state['refined_solution']}

Policies:
{TELECOM_POLICIES}

Give one final solution recommended in the last section, this solution will be directly implemented by the human agent
"""
    )
    return {"federated_solution": response.content}    
    
builder = StateGraph(SupportState)

builder.add_node("competitor_a", competitor_a)
builder.add_node("competitor_b", competitor_b)
builder.add_node("evaluator", evaluator)
builder.add_node("adversarial_agent", adversarial_agent)
builder.add_node("refiner_agent", refiner_agent)
builder.add_node("federated_agent", federated_agent)

# Sequential flow: each agent runs only after the previous one finishes.
builder.set_entry_point("competitor_a")

builder.add_edge("competitor_a", "competitor_b")
builder.add_edge("competitor_b", "evaluator")
builder.add_edge("evaluator", "adversarial_agent")
builder.add_edge("adversarial_agent", "refiner_agent")
builder.add_edge("refiner_agent", "federated_agent")
builder.add_edge("federated_agent", END)

app = builder.compile()


try:
    from IPython.display import Image
    img = Image(app.get_graph().draw_mermaid_png())
    with open(os.path.join(BASE_DIR, "assets", "telecom_agents.png"), "wb") as file:
        file.write(img.data)
except Exception as e:
    print(f"[Warning] Could not generate graph image: {e}")

result = app.invoke({
    "messages": [HumanMessage(content="Why was I charged international roaming despite disabling it?")],
    "solution_a": "",
    "solution_b": "",
    "selected_solution": "",
    "adversarial_feedback": "",
    "refined_solution": "",
    "federated_solution": ""
})

print("\n==============================")
print("FINAL FEDERATED SOLUTION\n")
print(result["federated_solution"])
print("==============================")
