# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import numpy as np
import datetime
import pytz
import os
# import functools
# import logging

# def console_logger():
#     # create console handler and set level to debug
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)

#     # create formatter
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     formatter = logging.Formatter('%(asctime)s %(funcName)s: %(message)s')

#     # add formatter to ch
#     ch.setFormatter(formatter)

#     # add ch to logger
#     logger = logging.getLogger('app')
#     logger.setLevel(logging.DEBUG)
#     logger.addHandler(ch)
#     return logger
# #logger = console_logger()from dataclasses import dataclass

# from dataclasses import dataclass
# @dataclass
# class SpreadTrade:

class OptionQuotes:
    CALL = 'CALL'
    PUT = 'PUT'
    def __init__(self, symbol,filename=None) -> None:
        self.symbol = symbol
        if filename:
            self.filename = filename
        else:
            self.filename = f'data/{symbol}.2023-05-04.parquet'
        self.last_mtime = 0
        self.reload(force=True)
        pass

    def pivot(self):
        if self._pivot is None:
            self._pivot = pd.pivot(self.data, index=['putCall', 'processDateTime'], columns=['strikePrice'], values=['mark', 'volume', 'delta'])
        return self._pivot

    def pivot_for(self, putCall, time, value):
        if time is None:
            time = pd.Timestamp.now().floor(freq='min')
        s = self.pivot().loc[(putCall, time), (value, slice(None))]
        return s.reset_index(name=value).drop(columns=['level_0'])

    def underlying_history(self):
        return self.data.groupby(['processDateTime']).underlyingPrice.mean()

    def reload(self, force=False) -> pd.DataFrame:
        """Returns Pandas Data Frame"""
        mtime = os.path.getmtime(self.filename)
        if force or (mtime > self.last_mtime):
            #logger.info(f'OptionQuotes reload {self.symbol}')
            self.last_mtime = mtime
            self.data = pd.read_parquet(self.filename)
            self.post_load_data()
#        else:
            #logger.info(f'OptionQuotes reload {self.symbol} skipped')
        self._pivot = None
        return self.data

    def post_load_data(self):
        df = self.data
        df.sort_values(['symbol', 'processDateTime'], inplace=True)

        vol = df['totalVolume'].diff()
        vol[df.symbol != df.symbol.shift(1)] = np.nan
        vol[vol < 5] = 0
        df['volume'] = vol

        s = df['mark'].diff()
        s[df.symbol != df.symbol.shift(1)] = np.nan
        df['mark_diff'] = s.round(2)
        df['netVolume'] = np.sign(df['mark_diff']) * df['volume']

        df['underlyingPrice'] = df.underlyingPrice.round(0)
        df['distance'] = (df['strikePrice'] - df['underlyingPrice']).apply(lambda x: round(x / 10) * 10)
        df['markVol'] = round(df.mark * df.volume,0)

        df['gexVol'] = (df.mark * df.volume * df.gamma).round(0)

        # df['sma5'] = df.mark.rolling(5).mean().round(2)
        # df['sma10'] = df.mark.rolling(10).mean().round(2)

        #time = df.processDateTime.dt.time
        #df.loc[(time >= pd.to_datetime('09:40:00').time() ),  ['sma5', 'sma10'] ] = np.nan

        df = self.filter_low_volume(df, 50)
        df = self.filter_rth(df)

        df.fillna(0, inplace=True)
        drops = ['tradeTimeInLong', 'quoteTimeInLong', 'netChange', 'gamma', 'rho', 'last',
            'bid', 'ask', 'highPrice', 'lowPrice', 'openPrice', 'closePrice', 'expirationDate', 'lastTradingDay', 'multiplier',
            'timeValue', 'theoreticalOptionValue', 'theoreticalVolatility', 'percentChange', 'markChange', 'markPercentChange', 'intrinsicValue',
        ]
        #for col in drops: df.pop(col)
        #df.pop([x for x in drops if x in df.columns])
        df.drop([x for x in drops if x in df.columns], inplace=True, axis=1)

        #strike_bins = pd.IntervalIndex.from_breaks(df.strikePrice.unique())
        #df['underlyingPriceBin'] = pd.cut(df.underlyingPrice, bins=strike_bins)
        #df = df[df.volume > 10]
        self.data = df

    def filter_rth(self, df=None):
        """filter the dataframe to remove rows before 9:30 today"""
        # df.loc[df['datetime'].dt.time < pd.to_datetime('9:30 AM').time(), 'spread1'] = 0

        if df is None: df = self.data
        time = df.processDateTime.dt.time
        return df[(time >= pd.to_datetime('09:30:00').time()) & (time <= pd.to_datetime('16:00:00').time())]
        #floor_dt = datetime.datetime(now_dt.year, now_dt.month, now_dt.day, 9, 30).astimezone(pytz.timezone('US/Eastern'))
        #floor_dt64 = np.datetime64(floor_dt)
        #ts = pd.Timestamp('today').floor('D') + datetime.timedelta(hours=9, minutes=30)
        #floor = datetime.fromtimestamp(ts)
        floor_dt = pytz.timezone("US/Eastern").localize(datetime.datetime.combine(datetime.datetime.now(), datetime.time(9, 30)))
        return df[df.processDateTime >= floor_dt]

    def filter_low_volume(self, df, minVolume):
        """
        # d0 = maDateTime.max()
        # keys = df[(maDateTime == d0) & (df.totalVolume < 500) ].symbol.unique()
        # df = df[ ~df['symbol'].isin(keys) ]
        Need group by symbol where max(totalVolume) < lowThreshold
        """
        s = df.groupby(['symbol']).totalVolume.max() < minVolume
        symbols = s[s].index.values
        df = df[ ~df['symbol'].isin(symbols) ]
        df = df.groupby('symbol').filter(lambda x: len(x) >= 2)
        return df.copy()

    def calc_spreads(self, df, distance):
        """ Given a list of [strikePrice, mark], return prices for spreads"""
        prices = df.dropna()
        spreads = df.dropna()

        spreads = spreads.rename(columns={'strikePrice': 'shortStrike', 'mark': 'shortPrice'})
        spreads['longStrike'] = spreads.shortStrike + distance
        spreads['distance'] = distance
        spreads['putCall'] = OptionQuotes.CALL

        spreads = pd.merge(spreads, prices, left_on='longStrike', right_on='strikePrice')
        spreads = spreads.rename(columns={'mark': 'longPrice'})
        spreads['price'] = round(spreads.shortPrice - spreads.longPrice,2)
        spreads.drop(columns=['shortPrice', 'longPrice', 'strikePrice'], inplace=True)
        if spreads.price.min() < 0 and spreads.price.max() <= 0.0:
            spreads.rename(columns={'shortStrike': 'longStrike', 'longStrike': 'shortStrike'}, inplace=True)
            spreads.price = -spreads.price
            spreads['putCall'] = OptionQuotes.PUT

        return spreads[spreads.price > 0.05]


    def find_spread(self, now: datetime, opts: dict) -> dict:
        """ opts is dict with putCall, distance, creditMin, creditTarget """
        df = self.pivot_for(opts['putCall'], now, 'mark')
        df = self.calc_spreads(df, opts['distance'])
        df['creditDiff'] = round(abs(opts['creditTarget'] - df.price),2)
        df = df[(df.price >= opts['creditMin'])]
        spread = df.sort_values('creditDiff').iloc[0]
        resp = opts.copy().update(spread.to_dict())
        resp = opts | spread.to_dict()
        resp['open_dt'] = now
        return resp
