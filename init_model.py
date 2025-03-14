from dotenv import  load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

def initialize_model():
    return ChatAnthropic(model="claude-3-5-sonnet-latest")