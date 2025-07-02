Dashboard for 0DTE opions

Designed to read a Pandas dataframe from a Parquet file if/when that file changes.
This approach allows the web server to not need/make calls to other Internet services.

The Pandas Dataframe is a slightly trimed result of essentially the following:
```curl --no-progress-meter -X GET "https://api.tdameritrade.com/v1/marketdata/chains?apikey=${TDA_CLIENT_ID}&symbol=%24SPX.X&strikeCount=50&fromDate=$d0&toDate=$d0" -o $filename
```
polling.py Demonstrates browser polling server to get incremental data
react.py

# TODOs
* The Plotly charts result in a total resend of the data. For the Pez dispenser, this is around 1mb. Ideally only the incremental data is sent. The browser can then figure out what to do after that.
* Until the incremental thing is fixd, the app uses extendable Graph, which maybe sucks.

# Daily
* Create csv for each parquet file
* gzip json file
* Concat SPX GEX files into single file

# Other Junk
gunicorn app:server -b :8050 --access-logfile access.log -D

Store
When used as input or state, all data is transfered to server from client. Same applies on Output.
A Client side script can manipulate the Store including setting it to Null. If set to Null in JS, Python server gets None


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
pattern = r'^(?P<ticker>\w+)_(?P<month>\d{2})(?P<day>\d{2})(?P<year>\d{2})(?P<putCall>[PC])(?P<strikePrice>\d+)$'
df[['ticker', 'month', 'day', 'year', 'putCall', 'strikePrice']] = df['symbol'].str.extract(pattern)


# Spyder
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from utils import OptionQuotes
filename ='../tda-tbd/wip/SPX.X.2023-06-15.GEX.parquet'
filename ='../tda-tbd/data/SPX.X.2023-06-28.parquet'
oq = OptionQuotes(symbol='abc',filename=filename)
df_base = oq.reload()
df = pd.read_parquet(filename)
dfx = df_base.loc[(df.processDateTime < pd.to_datetime('2023-06-22 14:59:00-04:00'))]
dfx = df.loc[(df.processDateTime == df.processDateTime.max())]

vix = pd.read_parquet('./vix.parquet')
df = pd.merge(df, vix.vix, left_index=True, right_index=True)

df['priceRet_quartile'] = df['priceRet'].fillna(0).abs().rolling(window=100).apply(
    lambda x: pd.qcut(x, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
)
df['priceRet_quartile'] = df['priceRet'].rolling(window=100).apply(calculate_quartiles)


def write_excel(df, filename):
    df = df.copy()
    for column in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, US/Eastern]']):
        df[column] = df[column].dt.tz_localize(None)
        #df['Dates'] = df['Dates'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.to_excel(filename, index=False)

def df_priorOpenInterest(df):
    max_dt = df.processDateTime.max()
    #from_dt = pd.to_datetime('2023-06-15 00:00-0400')
    #to_dt = pd.to_datetime('2023-06-17 00:00-0400')
    #df = df.loc[(df.processDateTime == max_dt) & (df.expirationDate >= from_dt) & (df.expirationDate < to_date)]
    df = df.loc[(df.processDateTime == max_dt)]
    return df[['symbol','openInterest']]

dfx = pd.read_parquet('../tda-tbd/wip/SPX.X.2023-06-14.GEX.parquet')
df_prior = dfx.loc[(dfx.processDateTime == dfx.processDateTime.max())][['symbol','openInterest']]

df_all = pd.read_parquet(filename)
df_all['gex'] = df_all.openInterest * df_all.gamma
df_all['priorOpenInterest'] = pd.merge(df_all[['symbol']], df_prior[['symbol', 'openInterest']], on='symbol', how='left')['openInterest']

today = datetime.datetime.now(pytz.timezone('US/Eastern'))
to_date = today + datetime.timedelta(days=90)
to_date = pd.to_datetime('2023-06-17', utc=True)
max_dt = df_all.processDateTime.max()
df = df_all.loc[(df_all.processDateTime == max_dt) & (df_all.expirationDate >= today) & (df_all.expirationDate < to_date) & (df_all.strikePrice >= 4100) & (df_all.strikePrice <= 4500)]


#df = df.groupby(['putCall', 'strikePrice']).gex.sum().to_frame()
#dfx = df.groupby(['strikePrice', 'putCall']).agg({'gex':'sum','underlyingPrice':'mean'})
#df = df.reset_index()
#daf = df.loc[(df.gex > 0.0)]

# data['sign'] = np.where(np.where(data['putCall'] == 'CALL', 1, -1))
dfx = df.loc[(df.putCall == 'PUT')]
putGex = (dfx.gex * dfx.strikePrice).sum() / dfx.gex.sum()
dfx = df.loc[(df.putCall == 'CALL')]
callGex = (dfx.gex * dfx.strikePrice).sum() / dfx.gex.sum()

dfx.loc[(dfx.putCall == 'PUT'), 'gex'] *= -1



import datetime
def load_data(filename):
    dfx = pd.read_parquet(filename)
    dfx['time'] = dfx.processDateTime.dt.time
    dfx.sort_values(['symbol', 'processDateTime'], inplace=True)
    dfx['underlyingPrice'] = dfx.underlyingPrice.round(0)
    dfx['distance'] = (dfx['strikePrice'] - dfx['underlyingPrice']).apply(lambda x: round(x / 5) * 5)
    gb = dfx.groupby('symbol')
    dfx['volume'] = gb['totalVolume'].diff().fillna(0)
    dfx = dfx.fillna(0)
    dfx['gex'] = dfx['volume'] * dfx.gamma
    dfx = dfx.loc[(dfx.processDateTime.dt.time < datetime.time(15, 30) )]
    dfx['distanceAbs'] = dfx.distance.abs()
    dfx = dfx.loc[(dfx.distance.abs() <= 100)]
    #dfx = dfx.loc[(dfx.processDateTime.dt.minute == 0) | (dfx.processDateTime.dt.minute == 30)]
    return dfx
dfx = load_data('wip/SPX.X.2023-07-10.chain.parquet')
