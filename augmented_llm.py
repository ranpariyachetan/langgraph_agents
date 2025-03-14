from init_model import initialize_model
from pydantic import BaseModel, Field

llm = initialize_model()

class SearchQuery(BaseModel):
    search_query: str = Field(description="Query that is optimized web search")
    justification: str = Field(description="Why this query is relevant to the user's request.")


structured_llm = llm.with_structured_output(SearchQuery)

output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")

print(output)

def multiply(a: int, b:int) -> int:
    return a * b

llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke("What is 2 time 5?")

print(msg.tool_calls)
print(msg)