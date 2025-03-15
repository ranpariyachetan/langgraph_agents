# Routing classifies an input and directs it to a specialized followup task.
# This workflow allows for separation of concerns, and building more specialized prompts.
# Without this workflow, optimizing for one kind of input can hurt performance on other inputs.
#
# When to use this workflow: Routing works well for complex tasks where there are distinct categories that are better handled separately,
# and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

from pydantic import BaseModel, Field
from init_model import initialize_model
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage

# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["poem", "joke", "story"] = Field(
        description="The next step in the routing process"
    )

llm = initialize_model()

router = llm.with_structured_output(Route)

# State
class State(TypedDict):
    input: str
    decision: str
    output: str

# Nodes
def llm_call_1(state: State):
    """Write a story"""

    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_2(state: State):
    """Write a joke"""

    result = llm.invoke(state["input"])
    return {"joke": result.content}

def llm_call_3(state: State):
    """Write a poem"""

    result = llm.invoke(state["input"])
    return {"poem": result.content}

def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    # Run the augmented LLM with structured output to serve as routing logic.
    decision = router.invoke(
        [
            SystemMessage(
                content= "Route the input to story, joke or poem based on user's request"
            ),
            HumanMessage(
                content=state["input"]
            )
        ]
    )

    return {"decision", decision.step}

# Conditional edge function to route to the appropriate node
def route_decision(state: State):

    # Return the node name you want to visit next
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"

# Build workflow
router_builder = StateGraph(State)

router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)

router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

# Compile Workflow
router_workflow = router_builder.compile()

# Show the workflow
display(router_workflow.get_graph().print_ascii())
