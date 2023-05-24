# Run this app with `python app.py` and
# gunicorn app:server -b :8050 --access-logfile access.log -D --reload
# visit http://127.0.0.1:8050/ in your web browser.
# https://realpython.com/python-dash/
# https://www.pythonanywhere.com/user/rcapozzi/
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime
import math
import pytz
import dash
from dash import Dash, html, dcc, Output, Input, dash_table, ctx
import dash_bootstrap_components as dbc
import logging
from utils import OptionQuotes
import dash_extendable_graph as deg
import glob
import re

def get_files():
    file_dict = {}
    #for filename in os.listdir('../tda-tbd/data/*.parquet'):
    for filename in glob.glob('../tda-tbd/data/*.parquet'):
        filename = filename.replace('\\', '/')
        match = re.search(r"([^/]+)\.parquet$", filename)
        if match:
            key = match.group(1)
            file_dict[key] = filename
    keys = sorted(file_dict.keys(), reverse=True)[:5]
    file_dict = {key: value for key, value in file_dict.items() if key in keys}
    return file_dict

def dash_layout():
    symbols = ['SPX.X', 'SPY']
    symbols = ['SPX.X']

    app.OptionQuotes = {}
    try:
        # for s in symbols:
        #     yyyymmdd = datetime.datetime.today().strftime('%Y-%m-%d')
        #     filename = f'data/{s}.{yyyymmdd}.parquet'
        #     if os.path.isfile(filename):
        #         app.OptionQuotes[s] = OptionQuotes(symbol=s,filename=filename)

        files = get_files()
        for k, v in files.items():
            # app.logger.info(f'files k={k} v={v}')
            app.OptionQuotes[k] = OptionQuotes(symbol=k,filename=v)
            app.OptionQuotes[k].data.file_label = k
        symbols += sorted(files.keys(), reverse=True)
        symbols = sorted(app.OptionQuotes.keys(), reverse=True)

    except Exception as e:
        return html.Div([
            html.Hr(),
            html.H1("You filthy Degen. Check back during market hours."),
            #html.Span(f'No file for {filename}'),
            html.Span(f'{e}', style={'padding': '5px', 'fontsize:': '10px'}),
        ])

    xfields = [ 'processDateTime']#, 'underlyingPrice', 'strikePrice']
    yfields = [ 'volume'] #, 'markVol', 'gexVol', 'mark', 'totalVolume', 'delta' ]

    return html.Div(
        children=[
            dbc.Alert(id='alerts'),
            #html.H1(children="SPX 0DTE Option Chain Analytics"),
            html.Hr(),
            html.Details([
                html.Summary('Secret Section', style={'color': 'red', 'background': 'black'}),
                html.Div(id="data-table-div"),
            ]),
            html.Div(id='metrics', style={'padding': '5px', 'fontsize:': '10px', 'font-family': 'monospace'},
                     children=dcc.Interval(id='graph-update', interval=60*1000)),
            html.Div(
                children=[
                    html.Div(children=[
                        html.Div(children="Symbol", className="menu-title"),
                        dcc.Dropdown(
                            id="symbol",
                            options=[ {"label": f,"value": f} for f in symbols ],
                            value=symbols[0],
                            clearable=False,
                            className="dropdown",
                            ),
                    ]),

                    html.Div(children=[
                        html.Div(children="x-axis", className="menu-title"),
                        dcc.Dropdown(
                            id="x-axis",
                            options=[ {"label": f,"value": f} for f in xfields ],
                            value=xfields[0],
                            clearable=False,
                            className="dropdown",
                            ),
                    ]),
                    html.Div(children=[
                        html.Div(children="y-axis", className="menu-title"),
                        dcc.Dropdown(
                            id="y-axis",
                            options=[ {"label": f,"value": f} for f in yfields ],
                            value=yfields[0],
                            clearable=False,
                            className="dropdown",
                        ),
                    ]),
                ],
                className="menu",
            ),
            html.Div(id='strikes-selector-div', className="card"),
            html.Div(dcc.Graph(id="pc-summary", config={"displayModeBar": False}), className="card",),
            html.Div(
                deg.ExtendableGraph(id="pc-volume-graph", config={"displayModeBar": False}, className="card")),
            dcc.Interval(id='pc-volume-interval', interval=60*1000),
            #html.Div(dcc.Graph(id="pc-put", config={"displayModeBar": False}), className="card",),
            # html.Div(children = dcc.Graph(id="scatter3d-call", config={"displayModeBar": False}), className="card", ),
            # html.Div(children = dcc.Graph(id="scatter3d-put", config={"displayModeBar": False}), className="card", ),
            #html.Div(id="data-table-div", className="card"),

        ]
    )

# ====================================================================


external_stylesheets = [{
    "href": ("https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap"),
    "rel": "stylesheet",
}]

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server
app.title = 'SPX 0DTE Chain React Analytics Peaker'
app.layout = dash_layout

@app.callback(
    Output("data-table-div", "children"),
    Input('symbol', 'value'),
    Input('graph-update','n_intervals'), prevent_initial_call=True
)
def table(symbol, n):
    now = app.OptionQuotes[symbol].data.processDateTime.max()
    data = app.OptionQuotes[symbol].data
    # df = data[(data.processDateTime == now) & (data.volume > data.volume.mean())]
    df = data[(data.processDateTime == now) & (data.totalVolume > data.totalVolume.mean())]
    dt = dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
    return [html.Label("Active Strikes"), dt]

@app.callback(
    Output("strikes-selector-div", "children"),
    Input('symbol', 'value'),
)
def update_strikesselector(symbol):
    """TODO suppress_callback_exceptions=True"""
    #logger.info(f"enter {symbol}")
    df = app.OptionQuotes[symbol].data
    df = df.loc[(df.totalVolume > 10)]
    min, max = df.strikePrice.min(), df.strikePrice.max()
    priceMin, priceMax = int(df.underlyingPrice.min()), int(df.underlyingPrice.max())
    priceRange = priceMax - priceMin
    if priceRange > 10 or priceMax > 1000:
        step_size = 25
        min = math.floor(min / step_size) * step_size
        max = math.floor(max / step_size) * step_size
        priceMin = math.floor(priceMin / step_size) * step_size
        priceMax = math.ceil(priceMax / step_size) * step_size
    else:
        step_size = 1

    return [
        html.Span("Strikes Selector", className='menu-title'),
        dcc.RangeSlider(min=min, max=max, step=step_size,
                marks={i: '{}'.format(i) for i in range(int(min), int(max), step_size)},
                value=[priceMin-step_size, priceMax + step_size],
                tooltip={"placement": "bottom", "always_visible": True}, id='strikes-rangeslider')
    ]

@app.callback([
        Output('pc-volume-interval', 'n_intervals'),
        Output('pc-volume-graph', 'figure'),
        Output('pc-volume-graph', 'extendData')],
        Input('pc-volume-interval', 'n_intervals'),
        Input("symbol", "value")
)
def func(n_intervals, symbol):
    ids = ctx.triggered_prop_ids
    # app.logger.info(f'pc-volume-graph << n={n_intervals} symbol={symbol} ids={ids}')
    # now_dt = datetime.datetime.now(pytz.timezone('US/Eastern'))
    if n_intervals is None or 'symbol.value' in ctx.triggered_prop_ids:
        n_intervals = 0
        fig = go.Figure(layout={'title':f'Cummulative Put/Call Volume {symbol}', 'template': 'plotly_dark', 'height':400},
            data=[go.Bar(name='Net', marker_color='lightslategray',x=[], y=[]), go.Scatter(name='SMA10', line_color="lightsalmon",x=[],y=[])])
    else:
        fig = dash.no_update

    yaxis = 'totalVolume'
    df = app.OptionQuotes[symbol].reload()

    s = df.groupby(['putCall', 'processDateTime'])[yaxis].sum()
    dfx = pd.DataFrame()
    dfx['puts'] = s.loc[('PUT', slice(None))]
    dfx['calls'] = s.loc[('CALL', slice(None))]
    dfx['net'] = dfx.calls - dfx.puts
    dfx['mean'] = dfx.net.rolling(10).mean()

    max_dt = datetime.datetime.utcfromtimestamp(n_intervals)
    max_dt = max_dt.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Eastern'))

    dfx = dfx[(dfx.index > max_dt)]
    if dfx.empty:
        return [n_intervals-1, fig, None]
    n_intervals = int(dfx.index.max().timestamp())

    data = [ { 'x': dfx.index, 'y': dfx['net'].values }, { 'x': dfx.index, 'y': dfx['mean'].values, }, ]
    #app.logger.info(f'pc-volume-graph >> n={n_intervals} max_dt={max_dt} data={data[-60:]}')       
    return [n_intervals, fig, data]

# Idealy return 
# {'x': '2023-05-19 09:56', 'y': [{'SPXW_051923P4250': -9 }, 'SPXW_051923C4250': 75 }] }
# { 'name': 'SPXW_051923P4250', 'x': datetime.datetime(2023, 5, 19, 9, 46), 'y': -9 },
# { 'name': 'SPXW_051923C4250', 'x': datetime.datetime(2023, 5, 19, 9, 46), 'y': 75 }
def chart_pc_summary(df, strikes, yaxis):
    """
    import plotly.io as pio
    pio.renderers.default='svg'
    value="%H:%M")
    """
    xaxis = 'processDateTime'
    hovertemplate = '<br>'.join(['%{fullData.name}', xaxis + '=%{x}', yaxis +'=%{y}', 'mark=%{customdata}', '<extra></extra>' ])
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    data = df[(df.strikePrice >= strikes[0]) & (df.strikePrice <= strikes[1])]
    data = data.sort_values(['symbol', 'processDateTime'])
    symbols = data.symbol.sort_values().unique()

    for s in symbols:
        sign = 1 if data[(data.symbol == s)].iloc[0].putCall == 'CALL' else -1
        x=data[(data.symbol == s)][xaxis].dt.tz_localize(tz=None)
        y=data[(data.symbol == s)][yaxis]
        maxVal =y.max()
        if maxVal < 100:
            continue
        y = y * sign
        cd=data[(data.symbol == s)].mark
        fig.add_trace(go.Scattergl(x=x, y=y, customdata=cd, name=s, text=s, mode='markers', hovertemplate=hovertemplate), secondary_y=False,)

    fig.add_trace(
        go.Scattergl(x=data[(data.symbol == symbols[0])][xaxis].dt.tz_localize(tz=None), y=data[(data.symbol == symbols[0])]['underlyingPrice'].values,
                   name="underlyingPrice",
                   marker_color='white'),
        secondary_y=True,
    )
    fig['layout']['yaxis2']['showgrid'] = False
    fig.update_yaxes(title_text="<b>underlyingPrice</b>", secondary_y=True)
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(title_text=f'SPX Call/Put Pez Dispenser {df._pretty_name} {yaxis}', template='plotly_dark', height=600)
    #x = datetime.datetime.now(pytz.timezone('US/Eastern'))
    #app.logger.info(f'chart_pc_summary returns x={x}')
    return fig

@app.callback(
    Output("pc-summary", "figure"),
    Input("symbol", "value"),
    Input("strikes-rangeslider", "value"),
    Input("x-axis", "value"),
    Input("y-axis", "value"),
    Input('graph-update','n_intervals'), prevent_initial_call=True
)
def update_charts(symbol, strikes, xaxis, yaxis, n):
    df = app.OptionQuotes[symbol].reload()
    if not hasattr(df, '_fig_summary'):
        #FIX: Issues UserWarning
        df._fig_summary = chart_pc_summary(df, strikes, yaxis)
    #fig_call.update_layout(height=600, width = 800)
    return df._fig_summary

def calc_interval_to_update():
    # now_dt = datetime.datetime.now(pytz.timezone('US/Eastern'))
    # maxProcessDateTime = df.processDateTime.max()
    # next_dt = maxProcessDateTime + datetime.timedelta(seconds=65)
    # seconds_delay = int((next_dt - now_dt).total_seconds())
    # interval = int(seconds_delay)
    # if now_dt.hour > 8 and now_dt.hour < 16:
    #     interval = seconds_delay if seconds_delay > 0 else 59
    # else: interval = 86400
    return 120

@app.callback( Output("metrics", "children"), [Input('symbol', 'value'), Input('graph-update','n_intervals')],)
def update_metrics(*args):
    symbol = args[0]
    style_metrics = {'padding': '5px', 'fontsize:': '10px'}
    oq = app.OptionQuotes[symbol]
    df = oq.data
    maxProcessDateTime = df.processDateTime.max()
    interval = 10

    return [
        html.Span(dcc.Interval(id='graph-update', interval=interval*1000)),
        html.Span(f"{maxProcessDateTime.strftime('%Y-%m-%d %H:%M:%S')}", style=style_metrics),
        html.Span(f"{symbol} Last: {df[df.processDateTime == maxProcessDateTime].underlyingPrice.min() }", style=style_metrics),
        html.Span(f"Range: {df.underlyingPrice.min()}/{df.underlyingPrice.max()}", style=style_metrics),
        html.Span(f"Strikes: {df.strikePrice.min()}/{df.strikePrice.max()}", style=style_metrics),
        html.Span(f"Refresh: {interval}s", style=style_metrics),
        # html.Div([
        #     html.Span(f"filename:{app.OptionQuotes.filename}", style=style_metrics),
        #     html.Span(f"size:{len(df)}", style=style_metrics),
        #     ]),
        ]

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
    print("Done")
