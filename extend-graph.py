"""
Extending from incremental, compare and contrast dash_extendable_graph with the standard Graph incremental update.
A store is added to extend-graph output to enable the client side JS to ...
"""
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
import dash_extendable_graph as deg

app = dash.Dash(__name__)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

fig = {'layout': {'title': 'Client Rendered Title', 'template':'plotly_dark', 'height':400}, 
       'data': [{'type': 'bar', 'name': 'MSFT', 'x':[], 'y':[]}, {'name': 'AAPL', 'x':[], 'y':[]}]}

app.layout = html.Div([
    html.Div([
        html.P("CS Graph Render from Browser"),
        dcc.Graph(id='cs-graph'),
    ]),

    html.Div([
        html.P("Extend Graph gets incremental data from server"),
        deg.ExtendableGraph(id='extend-graph', figure=fig),
    ]),

    dcc.Store(id='extend-store', data=None, modified_timestamp=0),
    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0),
])


app.clientside_callback(
    "function (data) { return updateGraph('cs-graph', data); }",
    Output('extend-store', 'data'),
    Input('extend-store', 'data'),
)

@app.callback(
        Output('extend-graph', 'extendData'),
        Output('extend-store', 'data', allow_duplicate=True),
        Input('interval-component', 'n_intervals'),
        State('extend-store', 'data'),
        prevent_initial_call=True
        # Input('extend-graph', 'extendData')
)
def func(n, data):
    """
    For a new trace, name is ok on the 1st call. The 2nd call wedges. Errors spew on the client.
    """
    app.logger.info(f'extend-graph  << n={n} data={data}')

    # Extend Graph allows list of dict
    data = [{
        'x': [n],
        'y': [random.randint(25, 45)],
    }]
    data.append({
        'x': [n],
        'y': [random.randint(50, 75)],
    })
    if n % 2 == 0:
        data.append({
            'x': [n],
            'y': [random.randint(80, 99)],
        })
    msg = {'ts': n, 'data': [[ 'MSFT', data[0]['y'][0]], ['AAPL', data[1]['y'][0] ]] }
    if n % 2 == 0:
        msg['data'].append(['TSLA', data[2]['y'][0]])
    app.logger.info(f'extend-graph  >> n={n} data={data}')
    app.logger.info(f'extend-graph  >> n={n}  msg={msg}')
    return data, msg


if __name__ == '__main__':
    app.logger.warning(__name__ + ' starting')
    app.run_server(debug=True, port=8050)

