Dashboard for 0DTE opions

Designed to read a Pandas dataframe from a Parquet file if/when that file changes.
This approach allows the web server to not need/make calls to other Internet services.

The Pandas Dataframe is a slightly trimed result of essentially the following:
```curl --no-progress-meter -X GET "https://api.tdameritrade.com/v1/marketdata/chains?apikey=${TDA_CLIENT_ID}&symbol=%24SPX.X&strikeCount=50&fromDate=$d0&toDate=$d0" -o $filename```

# TODOs
* The Plotly charts result in a total resend of the data. For the Pez dispenser, this is around 1mb. Ideally only the incremental data is sent. The browser can then figure out what to do after that.
* Until the incremental thing is fixd, the app uses extendable Graph, which maybe sucks.

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
filename ='../tda-tbd/data/SPX.X.2023-06-09.parquet'
oq = OptionQuotes(symbol='abc',filename=filename)
df_base = oq.reload()
df = pd.read_parquet(filename)
dfx = df.loc[(df.processDateTime >= pd.to_datetime('2023-06-06 15:59:00-04:00'))]

vix = pd.read_parquet('./vix.parquet')
df = pd.merge(df, vix.vix, left_index=True, right_index=True)

df['priceRet_quartile'] = df['priceRet'].fillna(0).abs().rolling(window=100).apply(
    lambda x: pd.qcut(x, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
)
df['priceRet_quartile'] = df['priceRet'].rolling(window=100).apply(calculate_quartiles)
