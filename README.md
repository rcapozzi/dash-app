gunicorn app:server -b :8050 --access-logfile access.log -D   

Store
When used as input or state, all data is transfered to server from client. Same applies on Output.
A Client side script can manipulate the Store including setting it to Null. If set to Null in JS, Python server gets None

Plotly
data [x:[], y:[], ...]

Server: Sends update message {
    ts: Timestamp, data: { key1: value1, key2: value2}
}
Server: Sends update message {
    data: [ Timestamp, [[ key1, value1 ], [key2, value2 ]], ]
}



#############
# https://dash.plotly.com/dash-core-components/graph#graph-properties
# https://stackoverflow.com/questions/65990492/what-is-the-correct-way-of-using-extenddata-in-dcc-graph
app.clientside_callback(
    """
    function (n_intervals) {
        return [{x: [n_intervals], y: [2]}, [0] ]
    }
    """,
    Output('extend-graph', 'extendData'),
    Input('interval-component', 'n_intervals')
)
#############

def update_options():
    pattern = r'^(?P<ticker>\w+)_(?P<month>\d{2})(?P<day>\d{2})(?P<year>\d{2})(?P<putCall>[PC])(?P<strikePrice>\d+)$'
    df[['ticker', 'month', 'day', 'year', 'putCall', 'strikePrice']] = df['symbol'].str.extract(pattern)
    df = pd.DataFrame({'processDateTime': [
        datetime.datetime(2023, 5, 19, 10, 0),
        datetime.datetime(2023, 5, 19, 10, 1),
        datetime.datetime(2023, 5, 19, 10, 2),
        datetime.datetime(2023, 5, 19, 10, 0),
        datetime.datetime(2023, 5, 19, 10, 1),
        datetime.datetime(2023, 5, 19, 10, 2)],
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT', 'MSFT'],
        'volume': [5, 10, 25, 105, 120, 140] })
    filename = '../tda-tbd/data/SPX.X.2023-05-19.parquet'
    df = pd.read_parquet(filename)
    oq = OptionQuotes(None,filename)
    df = oq.data
    dfx = df.pivot(index='processDateTime', columns='symbol', values='volume').reset_index()

    # Create the 'data' list
    # x_values = np.array(dfx['processDateTime'])
    x_values = dfx['processDateTime']
    data = []
    for symbol in dfx.columns[1:]:
        data.append({
            'name': symbol,
            'x': x_values,
            'y': dfx[symbol],
        })


@app.callback(
        [Output('pc-volume-interval', 'n_intervals'),
        Output('pc-volume-graph', 'extendData')],
        Input('pc-volume-interval', 'n_intervals'),
        Input("symbol", "value")
)
def func(n, symbol):
    app.logger.info(f'pc-volume-graph << n={n} symbol={symbol}')
    # now_dt = datetime.datetime.now(pytz.timezone('US/Eastern'))
    yaxis = 'totalVolume'
    df = app.OptionQuotes[symbol].reload()

    if n is None:
        app.logger.info(f'Initial load data')
        n = 0

    max_dt = datetime.datetime.utcfromtimestamp(n)
    max_dt = max_dt.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Eastern'))
    df = df[(df.processDateTime > max_dt)]
    if df.empty:
        app.logger.info(f'pc-volume-graph !! n={n} max_dt={max_dt} data=None')
        return [n-1, dash.no_update]
    n = df.processDateTime.max().timestamp()
    s = df.groupby(['putCall', 'processDateTime'])[yaxis].sum()

    puts = s.loc[('PUT', slice(None))]
    calls = s.loc[('CALL', slice(None))]
    net = calls - puts
    data = [{
        'x': calls.index,
        'y': net.values,
        #'type': 'bar',
    }]

    app.logger.info(f'pc-volume-graph >> n={n} max_dt={max_dt} data={data[-60:]}')
        
    return [n, data]


import pandas as pd
from utils import OptionQuotes
oq = OptionQuotes(symbol='abc',filename='../tda-tbd/data/SPX.X.2023-06-01.parquet')
df = oq.data


df.sort_values(['symbol', 'processDateTime'], inplace=True)
s = df['mark'].diff()
s[df.symbol != df.symbol.shift(1)] = np.nan
df['mark_diff'] = s

dfg = df.groupby('symbol').agg({'netVolume':'sum'})
dfg = dfg.rename(columns={'netVolume': 'cumNetVolume'})
df = pd.merge(df, dfg, on='symbol')
