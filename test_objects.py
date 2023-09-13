from langchain.tools import tool
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.callbacks.base import BaseCallbackManager
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
import pandas as pd
from langchain.chat_models import ChatOpenAI

memory = ConversationBufferMemory(memory_key="chat_history")
llm = ChatOpenAI(temperature=0, model="gpt-4")
data = pd.read_csv("test_data.csv")
python_func = PythonAstREPLTool(locals={"data": data})
python_tool  = Tool(
        name = "Python Executor",
        func=python_func.run,
        description="""
        Will Analyze the 'data' variable in the function
        Can execute python code. Will return python code snippet. 
        While returning python code snippet, it will only return python code and no other text.
        """
)

# tools = [python_tool]
# agent = initialize_agent(
#     tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory
# )


# response = agent.run("Create a variable a and equate it's value to 2.Save the value of 'a' in a .txt file")
# print(response)

response = llm.predict("""Create a figure in plotly express using the country column on the x axis and population on the y axis.
This should be a bar chart! 
Do not include any python function like fig.show()
Do not return anything else apart from the python code snippet required to generate this code!
""")
print(response)
# while True:
#     user_message = input("Enter your message \n")
#     agent.run(str(user_message))