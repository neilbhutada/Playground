# import re 
# response = """
# jhdkjdhkajhksajhd
# """

# print(response.split('```'))

# pattern = r'python\n(.*)'
# match = re.search(pattern, response, re.DOTALL)

# if match:
#     code = match.group(1)
#     pattern = r'(.*)```'
#     code = re.search( pattern, code , re.DOTALL).group(1)

# print(code)
# exec(code)

# fig.show()

# def random():
#     exec("a = 2", globals(), locals())
#     return locals()["a"]

# def random():
#         code = """ 
# import pandas as pd
# import plotly.express as px
# df = pd.read_csv('test_data.csv')
# fig = px.line(df, x='year', y='pop')
# a = 2
#         """
        
#         exec(code, globals(), locals()) 
#         fig_2 = locals()['fig']
#         return fig_2

# yo = random()
# print(yo)


a = 2

value = f"""
What did the dog say!
{a}
"""
print(value)
