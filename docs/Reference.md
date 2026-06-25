---
title: markmap
markmap:
  colorFreezeLevel: 2
  maxWidth: 250
  initialExpandLevel: 2
---

# Code
## CodeGen
### boilerplate
#### imports
```py

import os
import re
import getpass
from typing import Annotated, TypedDict, List

from pydantic import BaseModel, Field

from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_community.tools import DuckDuckGoSearchRun
```
#### set env

```py

def _set_env(var: str):
    if not os.environ.get(var):
        # .strip() removes any trailing newline/space that can otherwise end up
        # in the HTTP auth header and trigger a UnicodeEncodeError / 400.
        os.environ[var] = getpass.getpass(f"{var}: ").strip()

_set_env("GOOGLE_API_KEY")
```

#### llm

```py

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)

search_tool = DuckDuckGoSearchRun()
```

#### block pattern

```py

BLOCKED_PATTERNS = [
    r"os\.system",
    r"subprocess",
    r"shutil",
    r"socket",
    r"requests",
    r"__import__",
    r"eval\(",
    r"exec\(",
    r"open\(",
    r"rm\s+-rf",
]

BLOCKED_IMPORTS = [
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
]

def contains_blocked_content(text: str) -> bool:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
```

#### Code output
```py

class CodeOutput(BaseModel):
    prefix: str = Field(description="Explanation of approach")
    imports: str = Field(description="All required imports")
    code: str = Field(description="Executable code without imports")


# Structured LLM
code_llm = llm.with_structured_output(CodeOutput)
```

### building blocks
#### prompt
```py

code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an senior  expert LangChain LCEL engineer coding assistant. You will only generate  secure enterprise level code.

Use ONLY the provided documentation context to generate code.
Ensure:
- Latest import paths
- No deprecated APIs
- Compatible with LangChain >=0.3 and LangGraph >=0.2
- Fully executable

Sample Reference working code and statements found to be working:

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

import langchain


#  LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Tool
search = DuckDuckGoSearchRun()

# Create agent (NEW v1 API)
agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt="You are a helpful assistant."
)


print(response["messages"][-1].content)

Rules:
- NEVER use os, subprocess, socket, requests
- NEVER access filesystem
- NEVER use eval or exec
- Code must be safe and sandbox compatible
- Only modern LangChain >=0.3 imports

Provide:
1. Clear explanation
2. All required imports
3. Fully executable code
Ensure no missing variables."""
        ),
        ("placeholder", "{messages}"),
    ]
)

code_chain = code_gen_prompt | code_llm
```

#### graph state

```py

class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    docs: str
    generation: CodeOutput | None
    error: str
    iterations: int

MAX_ITERATIONS = 5
```

#### guards

```py

def input_guard(state: GraphState):
    user_input = state["messages"][-1].content

    if contains_blocked_content(user_input):
        raise ValueError("[BLOCKED] Unsafe user input detected.")

    return state

def output_guard(state: GraphState):
    solution = state["generation"]

    combined = solution.imports + "\n" + solution.code

    if contains_blocked_content(combined):
        print("[BLOCKED] Unsafe code detected in generation.")
        return {"error": "yes"}

    for blocked in BLOCKED_IMPORTS:
        if f"import {blocked}" in solution.imports:
            print("[BLOCKED] Blocked import detected.")
            return {"error": "yes"}

    return {"error": "no"}
```

#### fetch docs

```py

def fetch_docs(state: GraphState):
    print("[Search] Searching latest documentation...")

    query = state["messages"][-1].content

    search_query = f"""
    Use the reference link provided below to get latest documentation of Langchain:
    https://reference.langchain.com/python/langchain_core/documents/#langchain_core.documents.base.Blob

    Latest LangChain or LangGraph documentation examples, also if there are errors, search specifically on fixes for those errors:
    {query}
    """

    result = search_tool.run(search_query)

    # Strip script tags or HTML injection
    cleaned = re.sub(r"<script.*?>.*?</script>", "", result, flags=re.DOTALL)
    return {
        "docs": cleaned,
        "generation": state["generation"],
        "iterations": state["iterations"] + 1,
        "error": state["error"]
    }

```

#### generate
```py

def generate(state: GraphState):
    print("---- GENERATING CODE ----")

    result = code_chain.invoke({
        "messages": state["messages"]
    })

    return {
        "generation": result,
        "iterations": state["iterations"] + 1,
        "error": "no"
    }
```

#### Check node
```py

def check_code(state: GraphState):
    print("---- CHECKING CODE ----")

    solution = state["generation"]
    imports = solution.imports
    code = solution.code

    try:
      exec(imports)
    except Exception as e:
        print("Import failed:", e)
        return {
            "error": "yes",
            "messages": [
                HumanMessage(content=f"Import failed: {e}. Fix it.")
            ]
        }

    try:
      exec(imports + "\n"+ code)
    except Exception as e:
        print("Execution failed:", e)
        return {
            "error": "yes",
            "messages": [
                HumanMessage(content=f"Execution failed: {e}. Fix it.")
            ]
        }

    print("---- CODE VALID ----")
    return {"error": "no"}
```

#### decision
```py

def decide(state: GraphState):
    if state["error"] == "no":
        print("---- FINISHING ----")
        return END

    if state["iterations"] >= MAX_ITERATIONS:
        print("---- MAX ITERATIONS REACHED ----")
        return END

    print("---- RETRYING ----")
    return "fetch_docs"
```

### Core

#### build
```py

builder = StateGraph(GraphState)

builder.add_node("input_guard", input_guard)
builder.add_node("fetch_docs", fetch_docs)
builder.add_node("generate", generate)
builder.add_node("output_guard", output_guard)
builder.add_node("check_code", check_code)

builder.set_entry_point("input_guard")

builder.add_edge("input_guard", "fetch_docs")
builder.add_edge("fetch_docs", "generate")
builder.add_edge("generate", "output_guard")
builder.add_edge("output_guard", "check_code")
builder.add_conditional_edges("check_code", decide)

app = builder.compile()
```

#### draw

```py

from IPython.display import Image, display

img = Image(app.get_graph().draw_mermaid_png())

with open("graph_output.png", "wb") as file:
    file.write(img.data)
```

#### invoke
```py

question = """
Build a RAG chain using LCEL and FAISS.
Read data from a text file.
Load embeddings into FAISS.
Answer user questions using retrieved context.
"""

result = app.invoke({
	"messages": [HumanMessage(content=question)],
   "docs": "",
	"generation": None,
	"error": "no",
	"iterations": 0
})

print("\n\n========== FINAL OUTPUT ==========")
print("Explanation:\n", result["generation"].prefix)
print("\nImports:\n", result["generation"].imports)
print("\nCode:\n", result["generation"].code)
```
## Competing agents
### Bolier plate
#### imports
```py

import os
import getpass
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

```

#### env variable
```py

def _set_env(var: str):
    if not os.environ.get(var):
        # .strip() removes any trailing newline/space that can otherwise end up
        # in the HTTP auth header and trigger a UnicodeEncodeError / 400.
        os.environ[var] = getpass.getpass(f"{var}: ").strip()

_set_env("GOOGLE_API_KEY")
```

#### definig agents

```py

GEMINI_MODEL = "gemini-3.1-flash-lite"

llm_competitor = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)
llm_strategic  = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.2)
llm_evaluator  = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.0)
llm_adversary  = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.9)
llm_refiner    = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.0)
llm_federated  = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.0)
```

#### policies
```py

TELECOM_POLICIES = """
1. International roaming charges apply when customer device connects to foreign network.
2. Billing disputes must be resolved within 7 days.
3. Compensation up to 25% allowed for first-time billing errors.
4. Detailed itemized bill must be provided upon request.
"""

```
### Node creation
#### State
```py

class SupportState(TypedDict):
    messages: Annotated[List, add_messages]
    solution_a: str
    solution_b: str
    selected_solution: str
    adversarial_feedback: str
    refined_solution: str
    federated_solution: str

```
#### competitor
```py

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
```

#### eval node
```py

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
```

#### adversial node
```py

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
```

#### refiner agent
```py

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
```

#### federated agent

```py

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
```

### execution
#### edge creation
```py

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
```

#### draw
```py

from IPython.display import Image, display

img = Image(app.get_graph().draw_mermaid_png())

with open("graph_output_2.png", "wb") as file:
    file.write(img.data)
```

#### invoke

```py

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
```
## order processing
### boiler plate
#### imports
```py

import os
import getpass
import sqlite3
import uuid
import random
import re
from typing import TypedDict, Optional

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
```
#### env set
```py

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ").strip()

_set_env("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
```

#### extract text
```py

def extract_text(agent_resp) -> str:
    """Return the final agent message as a clean string.

    Gemini (and the LangChain 1.0 message format) can return message .content
    as a LIST of content blocks, e.g. [{"type": "text", "text": "..."}],
    instead of a plain string. Calling .strip() on a list raises
    AttributeError, so this normalizes both shapes to a string.
    """
    if isinstance(agent_resp, dict) and "messages" in agent_resp:
        content = agent_resp["messages"][-1].content
    else:
        return str(agent_resp).strip()

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()

    return str(content).strip()
```

#### extract order id
```py


_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

def extract_order_id(agent_resp) -> str:
    """Pull the bare order_id (UUID) out of the agent's reply.

    create_agent with Gemini tends to wrap the tool result in a full sentence
    ("...Your order ID is <uuid>."), so we extract just the UUID. Falls back to
    the full text only if no UUID is present.
    """
    text = extract_text(agent_resp)
    match = _UUID_RE.search(text)
    return match.group(0) if match else text
```

#### payment fail node
```py

def payment_failed(agent_resp) -> bool:
    """Reliably detect a payment failure.

    The payment tool returns a raw "FAIL:..." or "SUCCESS" string, but the LLM
    then paraphrases it ("The payment failed because..."), so checking only the
    final message text is unreliable. We scan ALL messages -- including the raw
    ToolMessage output -- for the failure signal.
    """
    if isinstance(agent_resp, dict) and "messages" in agent_resp:
        for msg in agent_resp["messages"]:
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            if "FAIL" in str(content).upper():
                return True
        return False
    return "FAIL" in extract_text(agent_resp).upper()
```
### Data handling
#### insert data
```py

conn = sqlite3.connect("orders.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    item TEXT,
    amount REAL,
    payment_status TEXT,
    notification_status TEXT
)
""")
conn.commit()
```
#### data contract
```py

class CreateOrderInput(BaseModel):
    item: str = Field(description="Item being ordered")
    amount: float = Field(description="Order amount")

class PaymentInput(BaseModel):
    order_id: str = Field(description="Order ID for payment")

class NotifyInput(BaseModel):
    order_id: str = Field(description="Order ID to notify")
```

#### payment and order
```py

def create_order_tool(data: CreateOrderInput):
    order_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
        (order_id, data.item, data.amount, "PENDING", "NOT_SENT")
    )
    conn.commit()
    return {"order_id": order_id}

def process_payment_tool(data: PaymentInput):
    cursor.execute(
        "UPDATE orders SET payment_status = ? WHERE order_id = ?",
        ("SUCCESS", data.order_id)
    )
    conn.commit()
    return {"payment_status": "SUCCESS"}

def notify_user_tool(data: NotifyInput):
    cursor.execute(
        "UPDATE orders SET notification_status = ? WHERE order_id = ?",
        ("SENT", data.order_id)
    )
    conn.commit()
    return {"notification": "Email Sent"}
```

#### tool creation
```py

@tool
def create_order_tool_agent(item: str, amount: float = 100.0) -> str:
    """Create an order and return the generated order_id"""
    result = create_order_tool(CreateOrderInput(item=item, amount=amount))
    return result["order_id"]

@tool
def process_payment_tool_agent(order_id: str) -> str:
    """Process payment for a given order_id"""
    result = process_payment_tool(PaymentInput(order_id=order_id))
    return result["payment_status"]


@tool
def notify_user_tool_agent(order_id: str) -> str:
    """Notify user for a given order_id"""
    result = notify_user_tool(NotifyInput(order_id=order_id))
    return result["notification"]
```
#### agent creation
```py

order_agent = create_agent(
    model=llm,
    tools=[create_order_tool_agent],
    system_prompt="""
You are an Order Service Agent. When given an item name and optional amount,
return the order_id string.
"""
)

payment_agent = create_agent(
    model=llm,
    tools=[process_payment_tool_agent],
    system_prompt="""
You are a Payment Service Agent. When given an order_id, return PAYMENT_SUCCESS or FAIL.
"""
)

notification_agent = create_agent(
    model=llm,
    tools=[notify_user_tool_agent],
    system_prompt="""
You are a Notification Service Agent. When given an order_id, return NOTIFICATION_SENT.
"""
)
```
### Core
#### state
```py

class WorkflowState(TypedDict):
    user_input: str
    order_id: Optional[str]
    payment_status: Optional[str]
    notification_status: Optional[str]
    error: Optional[str]
    retries: int

MAX_RETRIES = 2
```
#### processing node
```py

def order_requester(state: WorkflowState):
    print("[OrderRequester] Running")

    try:
        item = state["user_input"]
        amount = 100.0

        agent_resp = order_agent.invoke({
            "messages": [{"role": "user", "content": f"Create order for {item} with amount {amount}."}]
        })

        order_id = extract_order_id(agent_resp)

        return {
            "order_id": order_id,
            "retries": state["retries"]
        }

    except Exception as e:
        return {"error": str(e), "retries": state["retries"] + 1}

def payment_processor(state: WorkflowState):
    print("[PaymentProcessor] Running")

    try:
        agent_resp = payment_agent.invoke({
            "messages": [{"role": "user", "content": f"Process payment for order {state['order_id']}"}]
        })

        payment_status = extract_text(agent_resp)

        return {
            "payment_status": payment_status,
            "retries": state["retries"]
        }

    except Exception as e:
        return {"error": str(e), "retries": state["retries"] + 1}
```
#### notify node
```py

def notifier(state: WorkflowState):
    print("[Notifier] Running")

    try:
        agent_resp = notification_agent.invoke({
            "messages": [{"role": "user", "content": f"Notify user for order {state['order_id']}"}]
        })

        notification_status = extract_text(agent_resp)

        return {
            "notification_status": notification_status,
            "retries": state["retries"]
        }

    except Exception as e:
        return {"error": str(e), "retries": state["retries"] + 1}

# ==========================================================
# ERROR HANDLING & RETRY LOGIC
# ==========================================================

def check_error(state: WorkflowState):
    if state.get("error"):
        if state["retries"] >= MAX_RETRIES:
            print("[X] Max retries reached. Ending workflow.")
            return END
        print("[Retry] Retrying...")
        return "order_requester"
    return "payment_processor"
```
#### graph creation
```py

builder = StateGraph(WorkflowState)

builder.add_node("order_requester", order_requester)
builder.add_node("payment_processor", payment_processor)
builder.add_node("notifier", notifier)

builder.set_entry_point("order_requester")

builder.add_conditional_edges("order_requester", check_error)
builder.add_edge("payment_processor", "notifier")
builder.add_edge("notifier", END)

app = builder.compile()
```

#### invoke
```py

result = app.invoke({
    "user_input": "Laptop Purchase",
    "order_id": None,
    "payment_status": None,
    "notification_status": None,
    "error": None,
    "retries": 0
})

print("\n========== FINAL STATE (BASIC) ==========")
print(result)
```
