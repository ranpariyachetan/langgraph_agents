# In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.
#
# When to use this workflow: This workflow is well-suited for complex tasks where you can’t predict the subtasks needed
# (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task).
# Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined,
# but determined by the orchestrator based on the specific input."""

from typing import Annotated, List, TypedDict
import operator

from IPython.core.display_functions import display
from gandalf.metadata import description
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.utils.runnable import StrEnum
from pydantic import BaseModel, Field
from langgraph.constants import Send, START, END

from init_model import initialize_model

llm = initialize_model()

# Schema for structured output to use in planning
class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report."
    ),
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section"
    )

class Sections(BaseModel):
    sections : List[Section] = Field(
        description="Sections of the report"
    )

# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Sections)

# Graph state
class State(TypedDict):
    topic: str # Report topic
    sections: list[Section] # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]
    final_report: str

# Worker State
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

# Nodes
def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""
    # Generate queries
    report_sections = planner.invoke(
        [
            SystemMessage(
                content="Generate plan for the report."
            ),
            HumanMessage(
                content=f"Here is the report topic: {state['topic']}"
            )
        ]
    )

    return {'sections': report_sections.sections}

def llm_call(state: WorkerState):
    """Worker writer a section of report"""

    # Generate section
    section = llm.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description. Include not preamble for each section. Use markdown formatting."
            ),
            HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
            )
        ]
    )

    # Writer the updated section to completed sections.
    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    """Synthesize full report from sections"""
    # List of completed sections.
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections.
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}

# Conditional edge function to create llm_call workers that each writer a section of the report.
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel vai Send() API
    return[Send("llm_call", {"section": s}) for s in state["sections"]]

# Build Workflow
orchestrator_worker_builder = StateGraph(State)

orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)

orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# Compile the workflow
orchestrator_worker = orchestrator_worker_builder.compile()

# Show the workflow
display(orchestrator_worker.get_graph().print_ascii())

