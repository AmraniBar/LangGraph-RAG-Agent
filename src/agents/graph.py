from langgraph.graph import END, StateGraph, START
from agents.nodes import AgentState

def create_workflow(nodes):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Docs_Vector_Retrieve", nodes["retrieve"])
    workflow.add_node("Grading_Generated_Documents", nodes["grade_documents"])
    workflow.add_node("Content_Generator", nodes["generate"])

    workflow.add_edge(START, "Docs_Vector_Retrieve")
    workflow.add_edge("Docs_Vector_Retrieve", "Grading_Generated_Documents")
    workflow.add_edge("Grading_Generated_Documents", "Content_Generator")
    workflow.add_edge("Content_Generator", END)

    return workflow.compile()