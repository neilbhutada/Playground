import flask
import flask_login
from flask import Flask, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import dash
from dash import html

# Initialize Flask and Flask-Login
server = Flask(__name__)
server.secret_key = 'supersecretkey'

login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = 'login'

class User(UserMixin):
    # This is a basic user class; you'd typically be linking this to a database
    def __init__(self, id):
        self.id = id

# In a real scenario, this might query a database
users = {'username': {'password': 'password'}}

@server.route('/')
def index():
    return flask.redirect('/dash')
    


@server.route('/login', methods=['GET', 'POST'])
def login():
    # This is a very basic login view to demonstrate the process
    # In a real-world application, you'd validate the username and password, hash the password, etc.
    if flask.request.method == 'POST':
        username = flask.request.form.get('username')
        password = flask.request.form.get('password')
        if users.get(username) is not None and users[username].get('password') == password:
            user = User(id=username)
            login_user(user)
            return redirect('/')
    return '''
               <form method="post">
                   <input type="text" name="username" placeholder="Username">
                   <input type="password" name="password" placeholder="Password">
                   <input type="submit" value="Submit">
               </form>
           '''

@server.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

app1 = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix="/dash/",
    requests_pathname_prefix="/dash/"
)
app1.layout = html.Div("Hello, Dash app 1!")

# Protect the Dash app routes
@app1.server.before_request
def protect_dash_routes():
    if flask.request.path.startswith(app1.config['routes_pathname_prefix']):
        if not flask_login.current_user.is_authenticated:
            return redirect(flask.url_for('login'))

if __name__ == '__main__':
    server.run(debug=True, port=8050)
