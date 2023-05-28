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
from dash import Dash, html, dcc, Output, Input, State, dash_table, ctx
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
            html.Div(id='metrics-div', style={'padding': '5px', 'fontsize:': '10px', 'font-family': 'monospace'}, ),
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

            html.Div(deg.ExtendableGraph(id="pc-summary-graph", config={"displayModeBar": False}), className="card",),
            dcc.Interval(id='pc-summary-interval', interval=60*1000),
            dcc.Store(id="pc-summary-store", data=None, modified_timestamp=0),

            html.Div(deg.ExtendableGraph(id="pc-volume-graph", config={"displayModeBar": False}, className="card")),
            dcc.Interval(id='pc-volume-interval', interval=60*1000),
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
    # ids = ctx.triggered_prop_ids
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

def table_content(oq):
    now = oq.data.processDateTime.max()
    data = oq.data
    df = data[(data.processDateTime == now) & (data.totalVolume > data.totalVolume.mean())]
    dt = dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
    return [html.Label("Active Strikes"), dt]

def metric_content(symbol):
    style_metrics = {'padding': '5px', 'fontsize:': '10px'}
    oq = app.OptionQuotes[symbol]
    df = oq.data
    maxProcessDateTime = df.processDateTime.max()
    return [
        html.Span(f"{maxProcessDateTime.strftime('%Y-%m-%d %H:%M:%S')}", style=style_metrics),
        html.Span(f"{symbol} Last: {df[df.processDateTime == maxProcessDateTime].underlyingPrice.min() }", style=style_metrics),
        html.Span(f"Range: {df.underlyingPrice.min()}/{df.underlyingPrice.max()}", style=style_metrics),
        html.Span(f"Strikes: {df.strikePrice.min()}/{df.strikePrice.max()}", style=style_metrics),
    ]

def chart_pc_summary(df, strikes, yaxis, title=None):
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
    fig.update_layout(title_text=f'SPX Call/Put Pez Dispenser {title} {yaxis}', template='plotly_dark', height=600)
    state = { 'names': [row['name'] for row in fig["data"]],  'max_dt': df.processDateTime.max(), 'strikes': strikes, 'yaxis': yaxis }
    #app.logger.info(f'chart_pc_summary returns state={state}')
    #app.logger.info(f'chart_pc_summary returns x={x}')
    return state, fig


@app.callback(
    Output("pc-summary-store", "data", allow_duplicate=True),
    Output("pc-summary-graph", "extendData"),
    Output("data-table-div", "children"),
    Output("metrics-div", "children"), 
    Input('pc-summary-interval','n_intervals'), 
    State("pc-summary-store", "data"),
    prevent_initial_call=True
)
def func(n_interval, cookie):
    # grey = "\x1b[38;20m"
    # yellow = "\x1b[33;20m"
    # red = "\x1b[31;20m"
    # bold_red = "\x1b[31;1m"
    # reset = "\x1b[0m"
    # fmt = f'{yellow}extendData_cb{reset}'

    #dt = pytz.timezone("US/Eastern").localize(datetime.datetime(2023, 5, 19, 11, 0)) + datetime.timedelta(minutes=n_interval)
    max_dt = cookie['max_dt']
    #app.logger.info(f'{fmt} << n_interval={n_interval} {yellow}max_dt={max_dt}  {red}dt={dt}{reset}')

    symbol = cookie['symbol']
    oq = app.OptionQuotes[cookie['symbol']]
    df = oq.reload()
    #df = df[(df.processDateTime > max_dt) & (df.processDateTime <= dt)]
    df = df[(df.processDateTime > max_dt)]
    if df.empty:
        # app.logger.info(f'{fmt} >> No new data')
        return cookie, None, table_content(oq), metric_content(symbol)
    state, fig = chart_pc_summary(df, cookie['strikes'], cookie['yaxis'], title='Nope')

    data = {item['name']: {'x': item['x'], 'y': item['y']} for item in fig['data']}
    #app.logger.info(f'{fmt} >> data={data}')

    # 1. Every existing traces needs a row
    updates = []
    for name in cookie['names']:
        if name in data:
            updates.append(data[name])
        else:
            updates.append({'x':[], 'y':[]})
    # 2. Add new traces
    new_traces = set(data.keys()) - set(cookie['names'])
    for name in new_traces:
        updates.append(data[name])

    cookie['names'].extend(new_traces)
    cookie['max_dt'] = state['max_dt']
    # app.logger.info(f'{fmt} >> updates={updates}')
    c0 = table_content(oq)
    c1 = metric_content(symbol)
    return cookie, updates, c0, c1

@app.callback(
    Output("pc-summary-store", "data"),
    Output("pc-summary-graph", "figure"),
    Input("symbol", "value"),
    Input("strikes-rangeslider", "value"),
    Input("x-axis", "value"),
    Input("y-axis", "value"),
    prevent_initial_call=True
)
def func(symbol, strikes, xaxis, yaxis):
    df = app.OptionQuotes[symbol].reload()
    #dt = pytz.timezone("US/Eastern").localize(datetime.datetime(2023, 5, 19, 11, 0))
    #app.logger.info(f'pc_summary << dt={dt}')
    #dfx = df[(df.processDateTime < dt)]

    state, fig_summary = chart_pc_summary(df, strikes, yaxis, title=symbol)
    state['symbol'] = symbol
    return state, fig_summary


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
    print("Done")
