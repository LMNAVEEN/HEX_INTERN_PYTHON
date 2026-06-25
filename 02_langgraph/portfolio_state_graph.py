# Run: python 02_langgraph/portfolio_state_graph.py
# Tech: LangGraph + StateGraph
# Purpose: Simple LangGraph state machine — converts USD portfolio value to total with tax, then to INR

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from typing import TypedDict

class PortFolioState(TypedDict):
    amt_usd: float
    total: float
    total_inr: float

def calc_total(state:PortFolioState) -> PortFolioState:
    state['total'] = state['amt_usd'] * 1.08
    return state

def convert_to_inr(state:PortFolioState) -> PortFolioState:
    state['total_inr'] = state['total'] * 96
    return state    

from langgraph.graph import StateGraph,START,END

builder = StateGraph(PortFolioState)

builder.add_node("calc_tot", calc_total)
builder.add_node("conv_inr", convert_to_inr)

builder.add_edge(START, "calc_tot")
builder.add_edge("calc_tot","conv_inr")
builder.add_edge("conv_inr", END)


graph = builder.compile()

from IPython.display import Image, display

img = Image(graph.get_graph().draw_mermaid_png())

with open(os.path.join(BASE_DIR, "assets", "graph_output.png"), "wb") as file:
    file.write(img.data)

final_state = graph.invoke({"amt_usd":100})
print(final_state)    
