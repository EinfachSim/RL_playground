import requests
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Replace this with your actual API endpoint
API_URL = "http://127.0.0.1:8080/metrics"

app = dash.Dash(__name__)

# Layout with a graph and an interval component for live updates
app.layout = html.Div([
    html.H2("Live Rewards Dashboard"),
    dcc.Graph(id='reward-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1400,  # update every 2 seconds
        n_intervals=0
    )
])

# Callback to update the graph
@app.callback(
    Output('reward-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    try:
        response = requests.get(API_URL)
        data = response.json()
        episodes = data.get("episode", [])
        rewards = data.get("reward", [])
    except Exception as e:
        # In case of failure, show empty plot
        episodes, rewards = [], []

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes,
        y=rewards,
        mode='lines+markers',
        name='Reward'
    ))
    fig.update_layout(
        xaxis_title='Episode',
        yaxis_title='Reward',
        title='Average Rewards per Episode of training'
    )
    return fig

def run_dashboard_app(debug=False):
    app.run(debug=debug)
