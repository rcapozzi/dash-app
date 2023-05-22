import datetime
import pytz
import time
import dash
from dash.dependencies import ClientsideFunction, Input, Output, State
from dash import dcc, html
import random
import numpy as np
import pandas as pd
from flask import Flask, request

app = dash.Dash(__name__)
server = app.server  # Get the underlying Flask server

app.layout = html.Div([
    html.Div("Graph render with all data coming from server"),
    dcc.Graph(id='flask-graph'),
    html.Div("Extend Graph gets incremental data from server"),
    dcc.Graph(id='extend-graph'),
    html.Div("Client Side Callback Graph render with incremental data from server"),
    dcc.Graph(id='live-graph2'),
    dcc.Store(id='data-transfer', data=None, modified_timestamp=0),
    dcc.Store(id='data-store', data=None, modified_timestamp=0),
    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0),
    html.Script(src="/assets/customxxx.js"),
])

# Define a Flask route handler for the /update endpoint
@server.route('/update')
def update_data():
    # Retrieve the 'type' parameter from the request
    update_type = request.args.get('type')

    # Process the update based on the 'type' parameter
    incremental_data = []
    if update_type == 'example':
        # Example logic for generating incremental data
        incremental_data = [1, 2, 3]  # Replace with your logic

    # Return the incremental data as a JSON response
    return {'data': incremental_data}

@app.callback(Output('data-transfer', 'data'), Output('interval-component', 'n_intervals'), Input('interval-component', 'n_intervals'))
def func(n):
    """n_interval is set to the systems UTC by subtracting 1 to  ensure we get back the value we sent."""
    ts1 = int(time.time())
    app.logger.info(f'data-transfer << n={n}')
    if n == 0:
        app.logger.info('data-transfer initial load')
        x = ts1 - (5 * 10)
        initial_data = {'x': [x, x+10, x+20, x+30, x+40, ], 'y': [0, 10, 0, 10, 0]}
        dict = initial_data
    else:
        dict = { 'x': [ts1], 'y': [random.randint(1, 10)]}
    app.logger.info(f'data-transfer >> n={ts1} dict={dict}')
    return dict, ts1-1

app.clientside_callback(
    ClientsideFunction( namespace='clientside', function_name='merge_stores' ),
    Output('data-store', 'data'),
    Input('data-transfer', 'data'),
    Input('data-store', 'data'),
)

app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='do_graph1'),
    Output('live-graph2', 'figure'),
    Input('data-store', 'data'),
)


if __name__ == '__main__':
    app.logger.warning(__name__ + ' starting')
    app.run_server(debug=True, port=8050)

