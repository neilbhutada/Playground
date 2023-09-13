from langchain.tools import tool, StructuredTool, DuckDuckGoSearchRun
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent, initialize_agent, Tool
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

llm = VertexAI(temperature=0)

def query_data(filepath, query):

    query_agent = create_csv_agent(
        llm, 
        filepath,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=5
    )
     
    return query_agent.run(query)


class QueryCSV(BaseModel):
    filepath_and_query: str = Field(description="Should be a comma seperated list of form: file path, query")


@tool(args_schema=QueryCSV)
def query_csv(filepath_and_query:str) -> str: 
    """
    This tool will split the comma seperated input of filepath, query.
    After splitting the comma seperated input it will pass the filepath and query as a function to query_data
    The function will return analysis and answer query.
    """
    filepath, query = filepath_and_query.split(",")
    return  query_data(filepath, query)

search_func = DuckDuckGoSearchRun()
search_tool = Tool(
        name = "Current Search",
        func=search_func.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
)
    
tools = [query_csv, search_tool]
agent = initialize_agent(
    tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory
)


while True:
    user_message = input("Enter your message \n")
    agent.run(str(user_message))