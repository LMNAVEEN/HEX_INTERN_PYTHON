# Run: python 02_langgraph/code_gen_agent.py
# Tech: LangGraph + Groq (LLaMA) + DuckDuckGo Search + Structured Output
# Purpose: Agentic code generation loop with security guards, doc fetching, and self-correction

import os
import re
import getpass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from typing import Annotated, TypedDict, List

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field

from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_community.tools import DuckDuckGoSearchRun

def _set_env(var: str):
    if not os.environ.get(var):
        # .strip() removes any trailing newline/space that can otherwise end up
        # in the HTTP auth header and trigger a UnicodeEncodeError / 400.
        os.environ[var] = getpass.getpass(f"{var}: ").strip()

_set_env("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

search_tool = DuckDuckGoSearchRun()




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
    
class CodeOutput(BaseModel):
    prefix: str = Field(description="Explanation of approach")
    imports: str = Field(description="All required imports")
    code: str = Field(description="Executable code without imports")


# Structured LLM
code_llm = llm.with_structured_output(CodeOutput, method="json_mode")

code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a senior expert LangChain LCEL engineer. Generate secure, fully executable code using only modern LangChain >=0.3 APIs.

CORRECT modern import paths (use ONLY these):
- from langchain_groq import ChatGroq
- from langchain_huggingface import HuggingFaceEmbeddings
- from langchain_community.vectorstores import FAISS
- from langchain_community.document_loaders import TextLoader
- from langchain_text_splitters import RecursiveCharacterTextSplitter
- from langchain_core.prompts import ChatPromptTemplate
- from langchain_core.output_parsers import StrOutputParser
- from langchain_core.runnables import RunnablePassthrough

CORRECT LCEL RAG pattern:
retriever = vectorstore.as_retriever()
chain = (
    {{"context": retriever, "question": RunnablePassthrough()}}
    | prompt
    | llm
    | StrOutputParser()
)
answer = chain.invoke("user question here")

BANNED - never use these (they do not exist in LangChain >=0.3):
- LLMChain, RAGChain, load_faiss_index, QuestionAnswerer
- langchain.chains.rag, langchain.indexes
- create_agent from langchain.agents

Rules:
- NEVER use os, subprocess, socket, eval, exec
- Only modern LangChain >=0.3 imports listed above

You MUST respond with a valid JSON object with exactly these three keys:
{{"prefix": "<explanation string>", "imports": "<import statements as a single string>", "code": "<executable code without imports as a single string>"}}"""
        ),
        ("placeholder", "{messages}"),
    ]
)

code_chain = code_gen_prompt | code_llm

class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    docs: str
    generation: CodeOutput | None
    error: str
    iterations: int

MAX_ITERATIONS = 5



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

def decide(state: GraphState):
    if state["error"] == "no":
        print("---- FINISHING ----")
        return END

    if state["iterations"] >= MAX_ITERATIONS:
        print("---- MAX ITERATIONS REACHED ----")
        return END

    print("---- RETRYING ----")
    return "fetch_docs"
    

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


with open(os.path.join(BASE_DIR, "assets", "code_gen_flow.png"), "wb") as file:
    file.write(app.get_graph().draw_mermaid_png())

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
    