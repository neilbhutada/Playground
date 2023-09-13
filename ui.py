from dash import Dash, dcc, html, Input, Output, State, Patch, callback, MATCH
import dash_bootstrap_components as dbc
from langchain.tools import tool, StructuredTool, DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent, initialize_agent, Tool
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory
import sys
from io import StringIO
import re
import dash_uploader as du
import uuid
import shutil
import os
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.tools import ShellTool
import pandas as pd
import traceback



bot_answers = {}
bot_count = 0



def clean_text(text):
    # Replace any string of the form [<whatever text>m with "\n"
    cleaned = re.sub(r'(\x1b\[.*?m|\[.*?m)', '\n', text)
    
    # Insert a newline before the keywords "AI" and "Thought"
    cleaned = re.sub(r'(AI:|Thought:)', r'\n\1', cleaned)
    
    # Removing leading and trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned




memory = ConversationBufferMemory(memory_key="chat_history")

llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key = "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")



class createPlot(BaseModel):
    columns_and_plot_description_and_file_path: str = Field(description="""
    Should be dash '-' seperated list of form columns - plot description - filepath. 
    The columns will include the list of columns required for creating a plot.
    The plot description will include the description about the type of plot to be made using plotly express, 
    required data transformations, etc.
    The file path will include the file path of the dataset used for making the plot.
    """)

@tool(args_schema=createPlot)
def plot_creator(columns_and_plot_description_and_file_name:str) -> str: 
    """
    This tool will split the input into a - 'dash' seperated string with:
    1. Columns required for making the plotly express plot 
    2. The plot description for describing the kind of plot to be made, required data transformation, etc. 
    3. The filename of the dataset
    The input to this tool will be the input columns required for making the plot and plot description.
    The plot desciption can include the kind of plot, the required data transformations for making the plot, etc.
    The tool will generate the code required to make a plotly using plotly express with the given plot description
    and columns.
    """
    columns, plot_description, filepath = columns_and_plot_description_and_file_name.split("-")
    columns = columns.strip()
    plot_description = plot_description.strip()
    filepath = filepath.strip()
    filepath_updated=f'uploads/{unique_session}/{filepath}'
    print("The file path is ", filepath_updated)
    prompt = f"""
Use the columns: {columns}
filepath of the dataset: {filepath}
and the type of plot, required data transformations for creating the plot in the plot description: {plot_description}
to generate the code for creating a plot. Make sure to store the store the plotly express plot in 
a variable called 'fig'. Also store the list of columns in {columns} in a variable called 'cols'.
Read the dataset in a variable called 'df'.
Only have the python code and nothing else in the answer. Do not have 'fig.show()' in the python code
"""
    
    print("The prompt is \n", prompt)

    plot_code = llm.predict(prompt)

    return plot_code

def return_figure(match):
        print("I am here")
        code = match.group(1)
        pattern = r'(.*)`*'
        code = re.search( pattern, code , re.DOTALL).group(1)# Removing the delimiters
        code = code.split('```')[0]
        print(code)
        os.chdir(f'uploads/{unique_session}')
        exec(code, globals(), locals())   
        os.chdir('../..')
        return locals()['fig']



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
    If the user requests to save the resulting data frame after the analysis, the file must be saved as a .csv file.
    This tool can execute any python code required for Data Science purposes such as:
    training machine learning models, perform optimizations, perform A/B testing, create Monte Carlo based Simulations, etc.
    The tool will also execute shell commands - for example installing packages using pip - using the subprocess.run() function in python. 
    The tool will NOT be responsible for making any plots/visualizations and will NOT return any code about making plots.
    If this tool needs more context, it will not hesistate to ask for more context.
    The function will return analysis and answer question. 
    
    """
    filepath, question = filepath_and_question.split("-")
    filepath = filepath.strip()
    filepath=f"uploads/{unique_session}/{filepath}"
    question = question.strip()
    return  question_data(filepath, question)

search_func = DuckDuckGoSearchRun()
search_tool = Tool(
        name = "Current Search",
        func=search_func.run,
        description="""useful for when you need to answer questions about current events or the current state of the world. 
        Also useful for searching things up the internet.
        Useful for searching real-time information.
        """
)


    
tools = [question_csv, search_tool, plot_creator]
agent = initialize_agent(
    tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory
)



app = Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])
du.configure_upload(app, "uploads")
navbar = dbc.NavbarSimple(
    children=[
        # dbc.NavItem(dbc.NavLink("Page 1", href="#")),
        # dbc.DropdownMenu(
        #     children=[
        #         dbc.DropdownMenuItem("More pages", header=True),
        #         dbc.DropdownMenuItem("Page 2", href="#"),
        #         dbc.DropdownMenuItem("Page 3", href="#"),
        #     ],
        #     nav=True,
        #     in_navbar=True,
        #     label="More",
        # ),
    ],
    brand="Playground",
    brand_href="#",
    color="primary",
    dark=True,
)

def render_layout():
    global unique_session
    folder_path = 'uploads'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    unique_session = uuid.uuid1()
    layout = html.Div([
        navbar,
        html.Div([],id="chat-window", style={
        
        'overflow-y': 'scroll',
        'height': '20rem',
        'overflow-x': 'hidden'
    }),
    dcc.Loading(
    children = html.Div([], id= "output-loader", style={'height': '3rem'})
    ),
    html.Div(
    dbc.Row([
    dbc.Col(dbc.Textarea(id='chat-input', placeholder='Enter your message here...'), width=11),
    dbc.Col(dbc.Button('Send', id='send-button', style={
            'backgroundColor': '#333333',
            'borderColor': '#333333'} ), width=1, align="center"),
    ]),
    style = {'margin-bottom':'0.5rem'}
    ),
   dbc.Button("Upload File", outline=True, className="me-1", style={
            'borderColor': '#333333'}, id="open-upload"),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Upload File"), close_button=True),
                dbc.ModalBody(du.Upload(upload_id=unique_session))
            ],
            id="upload-box",
            centered=True,
            is_open=False,
        ),
        dcc.Download(id="download-results")
    ])
    return layout


app.layout = render_layout

@callback(
Output('chat-window', 'children', allow_duplicate=True),
# Input('chat-input', 'n_submit'),
Input('send-button', 'n_clicks'),
State('chat-input', 'value'),

  prevent_initial_call=True
)
def upload_user_question(clicked, text):
    card_user  = dbc.Col(
        dbc.Card(
        dbc.CardBody([
             html.P(
                text,
                className="card-text",
            )
        ]), color="light"),
        width={"size": 7}
        )
    messages = Patch()
    messages.extend([html.Br(), card_user])
    
    return messages


@callback(
Output('chat-window', 'children'),
Output('output-loader', 'children'), 
 Output("download-results", "data"),
# Input('chat-input', 'n_submit'),
Input('send-button', 'n_clicks'),
State('chat-input', 'value'),

  prevent_initial_call=True
)
def upload_bot_answer(clicked, text):
    global bot_answers, bot_count
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    if csv_files:
        os.remove(csv_files[0])
    original_stdout = sys.stdout
    sys.stdout = log_stream = StringIO()
    exception_occured = False
    try:
        card_color = "secondary"
        response = agent.run(str(text))
    except Exception as e:
        exception_occured = True
        card_color = "danger"
        exception_string = str(e)
        traceback_string = traceback.format_exc()
        response = html.Div([html.P(f"{exception_string}"),html.Br(),html.P(f"{traceback_string}")])
    sys.stdout = original_stdout
    log_stream.seek(0)
    bot_response = clean_text(log_stream.read())
    bot_response = bot_response.split('\n')
    bot_response_formatted = [html.P(line) for line in bot_response if line != ""]
    print("The bot count is ", bot_count)
    bot_answers[bot_count] = {
        'final_answer':response,
        'explanation': bot_response_formatted
    }
    pattern = r'python\n(.*)'
    match = re.search(pattern, response, re.DOTALL)
    changed = False
    if match and ("import plotly" in match.group(1)):
        try:
            figure_generated = return_figure(match)
            response = dcc.Graph(figure=figure_generated) 
            changed = True
            card_color = "secondary"
        except Exception as e:
            current_directory = os.getcwd()
            print(f"Current working directory: {current_directory}")
            os.chdir('../..')
            exception_occured = True
            card_color = "danger"
            exception_string = str(e)
            traceback_string = traceback.format_exc()
            response = html.Div([html.P(f"{exception_string}"),html.Br(),html.P(f"{traceback_string}")])

    bot_answers[bot_count] = {
        'final_answer':response,
        'explanation': bot_response_formatted
    }

    card_footer =   [dbc.Col(dbc.Button("Flip",
    style={'color':'#FFFFFF',
    'backgroundColor': '#333333',
    'borderColor': '#333333'}, 
    n_clicks=0, 
    id={'type':'flipper','index':bot_count}), width=1)
    ]

    csv_files = [f for f in os.listdir() if f.endswith('.csv')]

    # Check if any csv files are found
    if csv_files and (not exception_occured):
        results_download = pd.read_csv(csv_files[0])
        result_info = dbc.Col(html.Label(f"Check Downloads For Results",
    style={'color':'#FFFFFF','fontSize': '0.75rem'}), align="end")
        card_footer.append(result_info)


    card_body = [dbc.CardBody(response, id={'type':'bot_answer','index':bot_count}),
    dbc.CardFooter(
            dbc.Row(
            
            card_footer

            )
    )
    ]


    
    card_bot =  dbc.Col(
    dbc.Card(card_body,
    color = card_color,
    className="w-75 mb-3",
    style={
        'color':"#FFFFFF"
    }),
    width={"size": 8, "offset":6})
    messages = Patch()
    messages.extend([ html.Br(), card_bot])
    # html.Br(), card_user,
    bot_count+=1
    if csv_files and (not exception_occured):
        return messages, [], dcc.send_data_frame(results_download.to_csv, csv_files[0])
    else:
        return messages, [], None

@callback(
Output('chat-input', 'value'),
Input('send-button', 'n_clicks'),
  prevent_initial_call=True
)
def clear_input(clicked):
    return ""

@callback(
Output('upload-box', 'is_open'),
Input('open-upload', 'n_clicks'),

)
def open_upload_box(upload_button):
    if upload_button:
        return True

@callback(
    Output({'type':'bot_answer','index': MATCH}, 'children'),
    Input({'type':'flipper','index':MATCH}, 'n_clicks'),
    State({'type':'bot_answer','index':MATCH}, 'id'),
    prevent_initial_call=True
)
def flip_bot_answers(flips, id):
    answer_index = id['index']
    if flips%2 == 0:
        return bot_answers[answer_index]['final_answer']
    else:
        return bot_answers[answer_index]['explanation']
    



if __name__ == '__main__':
    app.run_server(debug=True)
