from langchain.tools import tool, StructuredTool, DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent, initialize_agent, Tool
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory
from langchain.tools.python.tool import PythonAstREPLTool

memory = ConversationBufferMemory(memory_key="chat_history")

llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key = "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")

def question_data(filepath, question):

    question_agent = create_csv_agent(
        llm, 
        filepath,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=5
    )
     
    return question_agent.run(question)


class questionCSV(BaseModel):
    filepath_and_question: str = Field(description="Should be a dash '-'seperated list of form: file path, question")


@tool(args_schema=questionCSV)
def question_csv(filepath_and_question:str) -> str: 
    """
    This tool will split the dash '-' seperated input of filepath, question.
    After splitting the dash '-'seperated input it will pass the filepath and question as a function to question_data
    The function will return analysis and answer question.
    """
    filepath, question = filepath_and_question.split("-")
    filepath = filepath.strip()
    question = question.strip()
    return  question_data(filepath, question)

search_func = DuckDuckGoSearchRun()
search_tool = Tool(
        name = "Current Search",
        func=search_func.run,
        description="useful for when you need to answer questions about current events or the current state of the world. Also useful for searching things up the internet."
)
    
tools = [question_csv, search_tool]
agent = initialize_agent(
    tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory
)


while True:
    user_message = input("Enter your message \n")
    agent.run(str(user_message))