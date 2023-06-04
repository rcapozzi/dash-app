# Run this app with `python app.py` and
# gunicorn app:server -b :8050 --access-logfile access.log -D --reload
# visit http://127.0.0.1:8050/ in your web browser.
# https://realpython.com/python-dash/
# https://www.pythonanywhere.com/user/rcapozzi/
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import glob
import re
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime
import math
import pytz
import dash
from dash import Dash, html, dcc, Output, Input, State, dash_table, ctx
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import dash_bootstrap_components as dbc
import dash_extendable_graph as deg
from utils import OptionQuotes

def seconds_to_monday():
    from datetime import datetime, timedelta, time
    eastern_tz = pytz.timezone('US/Eastern')
    try:
        current_time = datetime.now().astimezone(pytz.timezone('US/Eastern'))
        current_day = current_time.weekday()
        days_ahead = (0 - current_day + 7) % 7  # Number of days until next Monday
        next_monday = current_time + timedelta(days=days_ahead)
        target_time = datetime.combine(next_monday.date(), time(9, 30))
        target_time = eastern_tz.localize(target_time)
        time_difference = (target_time - current_time).total_seconds()
    except Exception as e:
        app.logger.info(f'seconds_to_monday ! {e}')
    return int(time_difference)

def is_market_open():
        from datetime import datetime, time
        current_time = datetime.now().astimezone(pytz.timezone('US/Eastern'))
        current_day = current_time.weekday()
        if (
            current_day >= 0 and current_day <= 4   # Monday to Friday
            and current_time.time() >= time(9, 30)         # After 9:30 am
            and current_time.time() <= time(16)            # Before or at 4 pm
        ):
            return True
        else:
            return False
def is_market_closed():
    return not is_market_open()

def calc_interval():
    if is_market_closed():
        return seconds_to_monday()
    return 60


def get_files():
    file_dict = {}
    for filename in glob.glob('../tda-tbd/data/*.parquet'):
        filename = filename.replace('\\', '/')
        match = re.search(r"([^/]+)\.parquet$", filename)
        if match:
            key = match.group(1)
            file_dict[key] = filename
    keys = sorted(file_dict.keys(), reverse=True)[:20]
    file_dict = {key: value for key, value in file_dict.items() if key in keys}
    return file_dict

def dash_layout():
    symbols = ['SPX.X', 'SPY']
    symbols = ['SPX.X']

    app.OptionQuotes = {}
    try:
        files = get_files()
        for k, v in files.items():
            app.OptionQuotes[k] = OptionQuotes(symbol=k,filename=v)
        symbols += sorted(files.keys(), reverse=True)
        symbols = sorted(app.OptionQuotes.keys(), reverse=True)
        app.OptionQuotes[symbols[0]].reload()
    except Exception as e:
        return html.Div([
            html.Hr(),
            html.H1("You filthy Degen. Check back during market hours."),
            html.Span(f'{e}', style={'padding': '5px', 'fontsize:': '10px'}),
        ])

    xfields = [ 'processDateTime', 'strikePrice']#, 'underlyingPrice', 'strikePrice']
    yfields = [ 'volume', 'markVol', 'distance', 'totalVolume', 'netVolume', 'mark'] #, 'markVol', 'gexVol', 'mark', 'totalVolume', 'delta' ]
    intervalDisabled = True # is_market_closed()
    interval = 60

    content = html.Div([
            dbc.Alert(id='alerts'),
            #html.H1(children="SPX 0DTE Option Chain Analytics"),
            html.Hr(),
            html.Details([
                html.Summary('Secret Section', style={'color': 'red', 'background': 'black'}),
                html.Div(id="data-table-div", children=table_content(app.OptionQuotes[symbols[0]])),
            ]),
            html.Div(id='metrics-div', style={'padding': '5px', 'fontsize:': '10px', 'font-family': 'monospace'}, ),
            dcc.Loading(
                html.Div([
                    dbc.Col(dcc.Graph(id="strike-volume", config={"displayModeBar": False}), style= {'width': '49%', 'display': 'inline-block'}, class_name='card'),
                    dbc.Col(dcc.Graph(id="strike-volume-right", config={"displayModeBar": False}), style= {'width': '49%', 'display': 'inline-block'}, className='card'),
                ]),
                type="cube"),

            dcc.Loading([
                html.Div(deg.ExtendableGraph(id="pc-summary-graph", config={"displayModeBar": False}), className="card"),
            ], type = 'default'),
            html.Div(children=[
                    html.Div(children=[
                        html.Div(children="Symbol", className="menu-title"),
                        dcc.Dropdown(
                            id="symbol",
                            options=[ {"label": f,"value": f} for f in symbols ],
                            value=symbols[0],
                            clearable=False, className="dropdown", ),
                    ]),
                    html.Div(children=[
                        html.Div(children="x-axis", className="menu-title"),
                        dcc.Dropdown(
                            id="x-axis",
                            options=[ {"label": f,"value": f} for f in xfields ],
                            value=xfields[0],
                            clearable=False, className="dropdown", ),
                    ]),
                    html.Div(children=[
                        html.Div(children="y-axis", className="menu-title"),
                        dcc.Dropdown(
                            id="y-axis",
                            options=[ {"label": f,"value": f} for f in yfields ],
                            value=yfields[0],
                            clearable=False, className="dropdown", ),
                    ]),
                ],
                className="menu",
            ),
            html.Div(id='strikes-selector-div', className="card"),
            html.Div([
                html.Button("Download Parquet", id="btn_parquet"),
                dcc.Download(id="download-dataframe-parquet"),
            ]),
            dcc.Interval(id='pc-summary-interval', interval=interval*1000, disabled=intervalDisabled),
            dcc.Store(id="pc-summary-store", data=None, modified_timestamp=0),
            html.Div(id="notify-container"),
            # html.Div(deg.ExtendableGraph(id="pc-volume-graph", config={"displayModeBar": False}, className="card")),
            # dcc.Interval(id='pc-volume-interval', interval=interval*1000, disabled=intervalDisabled),
        ])
    return dmc.MantineProvider(dmc.NotificationsProvider([content]))


# ====================================================================


external_stylesheets = [{
    "href": ("https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap"),
    "rel": "stylesheet",
}]

#TODO: use_pages=True
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server
app.title = 'SPX 0DTE Chain React Analytics Peaker'
app.layout = dash_layout

from flask import send_file, make_response
@app.server.route('/data/raw/<symbol>')
def serve_data_raw_file(symbol):
    app.logger.info(f'serve_data_raw_file {symbol}')
    oq = app.OptionQuotes[symbol]
    response = make_response(send_file(oq.filename))
    response.headers['Content-Disposition'] = f'attachment; filename=f"{symbol}.parquet"'
    return response

@app.server.route('/data/<symbol>')
def serve_data_file(symbol):
    import io
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    app.logger.info(f'{ts} serve_data_file {symbol}.parquet')
    oq = app.OptionQuotes[symbol]
    df = oq.reload()
    parquet_data = io.BytesIO()
    df.to_parquet(parquet_data)
    parquet_data.seek(0)
    response = make_response(send_file(parquet_data, mimetype='application/octet-stream',
                             as_attachment=True, download_name=f"{symbol}.parquet"))

    return response

@app.callback(
    Output("strikes-selector-div", "children"),
    Input('symbol', 'value'),
)
def update_strikesselector(symbol):
    """TODO suppress_callback_exceptions=True"""
    df = app.OptionQuotes[symbol].reload()
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

def chart_pc_summary(df, strikes, yaxis, xaxis, title=None):
    hovertemplate = '<br>'.join(['%{fullData.name}', xaxis + '=%{x}', yaxis +'=%{y}', 'mark=%{customdata}', '<extra></extra>' ])
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    data = df[(df.strikePrice >= strikes[0]) & (df.strikePrice <= strikes[1])]
    data = data.sort_values(['symbol', 'processDateTime'])
    symbols = data.symbol.sort_values().unique()

    for s in symbols:
        sign = 1 if data[(data.symbol == s)].iloc[0].putCall == 'CALL' else -1
        if xaxis == 'processDateTime':
            x=data[(data.symbol == s)][xaxis].dt.tz_localize(tz=None)
        else:
            x=data[(data.symbol == s)][xaxis]
        y=data[(data.symbol == s)][yaxis]
        y = y * sign
        cd=data[(data.symbol == s)].mark
        fig.add_trace(go.Scattergl(x=x, y=y, customdata=cd, name=s, text=s, mode='markers', hovertemplate=hovertemplate), secondary_y=False,)

    if xaxis == 'processDateTime':
        fig.add_trace(
            go.Scattergl(x=data[(data.symbol == symbols[0])][xaxis].dt.tz_localize(tz=None), y=data[(data.symbol == symbols[0])]['underlyingPrice'].values,
                    name="underlyingPrice",
                    marker_color='white'),
            secondary_y=True,
        )
    fig['layout']['yaxis2']['showgrid'] = False
    fig.update_yaxes(title_text="<b>underlyingPrice</b>", secondary_y=True)
    if xaxis == 'processDateTime':
        fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(title_text=f'SPX Call/Put Pez Dispenser {title} {yaxis}', template='plotly_dark', height=600)
    state = { 'names': [row['name'] for row in fig["data"]],  'max_dt': df.processDateTime.max(), 'strikes': strikes, 'yaxis': yaxis, 'xaxis': xaxis }
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
    state, fig = chart_pc_summary(df, cookie['strikes'], cookie['yaxis'], cookie['xaxis'], title='Nope')

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
    Output("notify-container", "children"),
    Output("pc-summary-interval", "disabled"),
    Output("pc-summary-store", "data"),
    Output("pc-summary-graph", "figure"),
    Input("symbol", "value"),
    Input("strikes-rangeslider", "value"),
    Input("x-axis", "value"),
    Input("y-axis", "value"),
    State("pc-summary-interval", "disabled"),
    prevent_initial_call=True
)
def func(symbol, strikes, xaxis, yaxis, intervalDisabled):
    # notification = None
    # app.logger.info(f'pc_summary << symbol={symbol}')
    # if 'symbol.value' in ctx.triggered_prop_ids:
    date_string = symbol.split(".")[-1]
    today = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
    intervalDisabled = False if date_string == today and is_market_open() else True
    s = "disabled" if intervalDisabled else "enabled"
    icon="ic:round-celebration"
    icon="mdi:gun"
    notification = dmc.Notification(id="my-notification", message=f"Updates {s} for {symbol}", color="green", action="show", icon=DashIconify(icon=icon),autoClose=5_000)
    df = app.OptionQuotes[symbol].reload()
    #dt = pytz.timezone("US/Eastern").localize(datetime.datetime(2023, 5, 19, 11, 0))
    #dfx = df[(df.processDateTime < dt)]
    state, fig_summary = chart_pc_summary(df, strikes, yaxis, xaxis, title=symbol)
    state['symbol'] = symbol
    return notification, intervalDisabled, state, fig_summary

@app.callback(
    Output("download-dataframe-parquet", "data"),
    Input("btn_parquet", "n_clicks"),
    State("symbol", "value"),
    prevent_initial_call=True,
)
def func(n_clicks, symbol):
    """ dcc.send_data_frame(df.to_parquet, f"symbol.parquet") """
    oq = app.OptionQuotes[symbol]
    #df = app.OptionQuotes[symbol].reload()
    return dcc.send_file(oq.filename)

@app.callback(
    Output("strike-volume", "figure"),
    Output("strike-volume-right", "figure"),
    Input('pc-summary-interval','n_intervals'),
    Input("symbol", "value"),
)
def func(n, symbol):
    df = app.OptionQuotes[symbol].reload()
    df = df.sort_values(['symbol', 'processDateTime'])

    df['sma5'] = df.volume.rolling(5).mean().round(2)
    df['sma15'] = df.volume.rolling(15).mean().round(2)
    df['vwap5'] = df.groupby('symbol').apply(calculate_vwap, window=5).values
    df['vwap15'] = df.groupby('symbol').apply(calculate_vwap, window=15).values

    max_dt = pd.to_datetime('2023-05-31 11:30:00-04:00')
    max_dt = df.processDateTime.max()
    dt = max_dt.strftime('%Y-%m-%d %H:%M')

    # dfg = df.groupby('symbol').agg({'netVolume':'sum'})
    # dfg = dfg.rename(columns={'netVolume': 'cumNetVolume'})
    # df = pd.merge(df, dfg, on='symbol')

    df.loc[df['putCall'] == 'CALL', 'totalVolume'] *= -1
    df.loc[df['putCall'] == 'CALL', 'volume'] *= -1
    df.loc[df['putCall'] == 'CALL', 'sma5'] *= -1
    df.loc[df['putCall'] == 'CALL', 'sma15'] *= -1
    df_base = df.copy()
    df = df[(df.processDateTime == max_dt)]
    underlyingPrice = df.underlyingPrice.abs().max()

    fig = go.Figure(layout=go.Layout(title=go.layout.Title(text=f"Total Volume for Twoday {dt}"), barmode='overlay'))
    fig.update_layout(barmode='overlay', yaxis_title='Strike Price', )
    fig.update_layout(legend=dict(yanchor="bottom", y=1.05, xanchor="right", x=1, orientation="h",))
    fig.update_yaxes(autorange="reversed")

    fig.add_vline(x=0, line_color='black')
    fig.add_hline(y=underlyingPrice, line_color='crimson', line_dash='dot', annotation_text=f'SPX {int(underlyingPrice)}')

    xaxis = 'totalVolume'
    puts = df[(df.putCall == 'PUT')]
    calls = df[(df.putCall == 'CALL')]
    fig.add_trace(go.Bar(x=calls[xaxis], y=calls.strikePrice, name='calls', orientation='h', marker_color='rgb(26, 118, 255)', ))
    fig.add_trace(go.Bar(x=puts[xaxis], y=puts.strikePrice, name='puts', orientation='h', marker_color='rgb(55, 83, 109)', ))
    # NOTE: Does not work with reversed.
    # ymax = max(calls[xaxis].abs().max(), puts[xaxis].abs().max() )
    # fig.update_yaxes(range=[-ymax, ymax])

    # Figure 2
    fig2 = go.Figure(layout=go.Layout(title=go.layout.Title(text=f"Prior One Minute Volume (mark over 0.50 & sma5 vol > 10) {dt}"), barmode='overlay'))
    fig2.update_layout(legend=dict(yanchor="bottom", y=1.05, xanchor="right", x=1, orientation="h",), template='plotly_dark')
    fig2.update_yaxes(autorange="reversed")

    fig2.add_vline(x=0, line_color='yellow')
    fig2.add_hline(y=underlyingPrice, line_color='crimson', line_dash='dot', annotation_text=f'SPX {int(underlyingPrice)}')

    m0 = (df.volume > df.sma5) & (df.sma5 > df.sma15) & (df.volume > 0) & (df.mark > 0.25)
    m1 = (df.volume < df.sma5) & (df.sma5 < df.sma15) & (df.volume < 0) & (df.mark > 0.25)
    dfx = df.loc[(m0) | (m1)]
    for index, row in dfx.iterrows():
        action = 'Buy' if row.markDiff > 0 else 'Sell'
        fig2.add_annotation(text=f"{action} {row.strikePrice:.0f}@{row.mark:.2f}", x=row.volume, y=row.strikePrice, arrowhead=1, showarrow=True)

    df_base_mask = (df_base.mark > 0.44) & (df_base.sma5.abs() > 10)
    unique_dates = df_base['processDateTime'].sort_values().unique().tolist()[-5:]
    frames = []
    for max_dt in unique_dates:
        data = []
        dt = max_dt.strftime('%Y-%m-%d %H:%M')
        df = df_base[(df_base.processDateTime == max_dt) & df_base_mask]
        puts = df[(df.putCall == 'PUT')]
        calls = df[(df.putCall == 'CALL')]
        data.append(go.Bar(x=calls.volume, y=calls.strikePrice, name='calls', orientation='h', marker_color='rgb(26, 118, 255)', ))
        data.append(go.Bar(x=puts.volume, y=puts.strikePrice, name='puts', orientation='h', marker_color='rgb(55, 83, 109)', ))
        data.append(go.Bar(x=df.sma5, y=df.strikePrice, name='sma5', orientation='h', width=1, marker_color='indianred', showlegend=True,))
        data.append(go.Scatter(x=df.sma15, y=df.strikePrice, name='sma15', mode='markers', orientation='h',
                marker=dict(size=12, symbol="line-ns", line=dict(width=2, color="pink"))
            ))
        frames.append({'data': data, 'name': dt})

    [fig2.add_trace(trace) for trace in frames[-1]['data']]
    dict2 = fig2.to_dict()
    xaxis = 'volume'
    xmax = max(calls[xaxis].abs().max(), puts[xaxis].abs().max() )
    xmax = math.ceil(xmax / 100) * 100 + 100
    fig.update_yaxes(range=[-xmax, xmax])
    dict2['layout']['xaxis'] = {"range": [-xmax, xmax], 'title': xaxis }
    dict2['layout']['updatemenus'] = [dict(type="buttons", font={'color':'black'}, buttons=[dict(label="last5", method="animate", args=[None])])]
    dict2['frames'] = [ f for f in frames ]
    fig2 = go.Figure(dict2)
    # app.logger.info(f'Made fig {datetime.datetime.now()}')

    return fig, fig2

def calculate_vwap(data, window=10):
    rolling_pv = (data['volume'] * data.mark).rolling(window=window, min_periods=1).sum()
    rolling_volume = data['volume'].rolling(window=window,min_periods=1).sum()
    vwap = rolling_pv / rolling_volume
    vwap.rename('vwap', inplace=True)
    vwap[pd.isna(vwap)] = data.mark
    return vwap

# import logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# class TimestampedHandler(logging.Handler):
#     def emit(self, record):
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         record.msg = f"{timestamp} - {record.msg}"
#         super().emit(record)

# logging.getLogger().addHandler(TimestampedHandler())

if __name__ == '__main__':
    app.logger.info("Dash app starting")
    app.run_server(debug=True, host='0.0.0.0')
    print("Done")
