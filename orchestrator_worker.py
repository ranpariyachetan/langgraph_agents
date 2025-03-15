# In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.
#
# When to use this workflow: This workflow is well-suited for complex tasks where you can’t predict the subtasks needed
# (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task).
# Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined,
# but determined by the orchestrator based on the specific input."""

from typing import Annotated, List
import operator

from pydantic import BaseModel, Field

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