# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import numpy as np
import datetime
import pytz
import os

class OptionQuotes:
    CALL = 'CALL'
    PUT = 'PUT'
    def __init__(self, symbol,filename=None) -> None:
        self.cache = {}
        self.symbol = symbol
        if filename:
            self.filename = filename
        else:
            self.filename = f'data/{symbol}.2023-05-04.parquet'
        self.last_mtime = 0
        self.data = None
        # self.reload(force=True)
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
            self.cache = {}
            self.post_load_data()
            self.max_dt = self.data.processDateTime.max()
            self.cache_set('max_dt', self.data.processDateTime.max())
#        else:
            # print(f'OptionQuotes reload {self.symbol} skipped', flush=True)
        self._pivot = None
        return self.data
    def cache_set(self,key,value):
        self.cache[key] = value
        return value
    def cache_get(self,key):
        return self.cache.get(key,None)

    def post_load_data(self):
        df = self.data
        for field in ['volatility', 'delta', 'gamma', 'theta']:
            df.loc[(df[field] < 0), field] = df.iloc[1][field]

        df.sort_values(['symbol', 'processDateTime'], inplace=True)

        gb = df.groupby('symbol')
        df['volume'] = gb['totalVolume'].diff().fillna(0)
        df['markDiff'] = gb['mark'].diff().fillna(0).round(4)
        df['markPctChange'] = gb['mark'].pct_change().fillna(0)
        df['underlyingPriceDiff'] = gb['underlyingPrice'].diff().fillna(2)


        df['upDown'] = np.sign(  gb['mark'].diff() )
        for i in range(2,10):
            df.loc[(df.upDown == 0) & (df.mark > 0.19), 'upDown' ] = np.sign(  gb['mark'].diff(i) )
        df['volumeUpDown'] = df['upDown'] * df['volume']

        df.loc[(df.markDiff == 0), 'volumeUpDown'] = np.sign(df['underlyingPriceDiff']) * df['volume']

        # df['volumeUpDownCum'] = df.groupby('symbol').apply(lambda x: x['volumeUpDown'].cumsum()).values
        df['volumeUpDownCum'] = df.groupby('symbol').volumeUpDown.cumsum()
        df['openInterestNet'] = df.openInterest + df['volumeUpDownCum']
        df['gex'] = df['openInterestNet'] * df.gamma

        df['underlyingPrice'] = df.underlyingPrice.round(0)
        df['distance'] = (df['strikePrice'] - df['underlyingPrice']).apply(lambda x: round(x / 10) * 10)

        # df['sma5'] = df.mark.rolling(5).mean().round(2)
        # df['sma10'] = df.mark.rolling(10).mean().round(2)

        #time = df.processDateTime.dt.time
        #df.loc[(time >= pd.to_datetime('09:40:00').time() ),  ['sma5', 'sma10'] ] = np.nan

        df = self.filter_low_volume(df, 50)
        df = self.filter_rth(df)

        df.fillna(0, inplace=True)
        drops = ['tradeTimeInLong', 'quoteTimeInLong', 'netChange', 'rho', 'vega',
            'bid', 'ask', 'highPrice', 'lowPrice', 'openPrice', 'closePrice', 'expirationDate', 'lastTradingDay', 'multiplier',
            'timeValue', 'theoreticalOptionValue', 'theoreticalVolatility', 'percentChange', 'markChange', 'markPercentChange', 'intrinsicValue',
            'upDown', 'volumeUpDown', 'volumeUpDownCum',
        ]
        df.drop([x for x in drops if x in df.columns], inplace=True, axis=1)

        #strike_bins = pd.IntervalIndex.from_breaks(df.strikePrice.unique())
        #df['underlyingPriceBin'] = pd.cut(df.underlyingPrice, bins=strike_bins)
        #df = df[df.volume > 10]
        self.data = df

    def filter_rth(self, df=None):
        """filter the dataframe to remove rows before 9:30 today"""
        if df is None: df = self.data
        time = df.processDateTime.dt.time
        return df[(time >= pd.to_datetime('09:30:00').time()) & (time <= pd.to_datetime('16:00:00').time())]

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

class EasternDT:
    utc_timezone = pytz.timezone('UTC')
    eastern_timezone = pytz.timezone('US/Eastern')

    @classmethod
    def u2e(cls, unix_timestamp=0):
        if unix_timestamp > 2**32:
            unix_timestamp = unix_timestamp/1000
        if unix_timestamp is None: unix_timestamp = 0
        utc_datetime = datetime.datetime.utcfromtimestamp(int(unix_timestamp))
        return cls.utc_timezone.localize(utc_datetime).astimezone(cls.eastern_timezone)

    @classmethod
    def e2u(cls, eastern_datetime):
        if eastern_datetime.tzinfo is None:
            eastern_datetime = cls.eastern_timezone.localize(eastern_datetime)
        utc_datetime = eastern_datetime.astimezone(cls.utc_timezone)
        return int(utc_datetime.timestamp())

    @classmethod
    def now(cls):
        return datetime.datetime.now(cls.eastern_timezone)

def tos_ts_0dte(symbol='SPY'):
    """Thinkscript for SPY 0DTE"""
    import yfinance as yf
    from jinja2 import Template
    from datetime import datetime, timedelta

    yf_symbol = '^GSPC' if symbol == 'SPX' else symbol
    symbol = 'SPXW' if symbol == 'SPX' else symbol

    # Fetch SPY price from Yahoo Finance
    quote = yf.Ticker(yf_symbol)
    price = quote.history(period="1d").iloc[-1]['Close']

    if price > 1000:
        price = round(price / 5) * 5
        # ary = [0, 20, 25, 40, 50, 60, 75, 80]
        strike_prices = [int(price - i) for i in range(100, -106, -5)]
    else:
        strike_prices = [int(price - i) for i in range(5, -6, -1)]

    # Get current date and time
    expiration_date = datetime.now()
    expiration_date = EasternDT.now()
    if expiration_date.hour >= 16:
        expiration_date = expiration_date + timedelta(days=1)
    else:
        expiration_date = expiration_date
    expiration_date = expiration_date.strftime("%y%m%d")

    # Generate option codes
    call_codes = [f"{symbol}{expiration_date}C{price}" for price in strike_prices]
    put_codes = [f"{symbol}{expiration_date}P{price}" for price in strike_prices]

    # Create Jinja2 template
    template_code = '''
declare lower;
input fundamentalType = FundamentalType.OHLC4;
script fixnan{
    input source = close;
    def fix = if !isNaN(source) then source else fix[1];
    plot result = fix;
}
script fixnanMultiply{
    input s1 = close;
    def fix1 = if !isNaN(s1) then s1 else fix1[1];
    input s2 = close;
    def fix2 = if !isNaN(s2) then s2 else fix2[1];
    plot result = fix1 * fix2;
}

def callMetric =
    {% for option in call_codes %}(fixnan(volume(".{{ option }}")) * fixnan(Fundamental(fundamentalType,".{{ option }}"))) +
    {% endfor %} 0;
def putMetric =
    {% for option in put_codes %}(fixnanMultiply(volume(".{{ option }}"), Fundamental(fundamentalType,".{{ option }}"))) +
    {% endfor %} 0; # Trailing zero saves the day

plot Calls = if SecondsFromTime(935) >= 0 then callMetric else 0;
Calls.SetPaintingStrategy(PaintingStrategy.SQUARED_HISTOGRAM);
Calls.AssignValueColor(Color.UPTICK);

plot Puts = if SecondsFromTime(935) >= 0 then -1.0 * putMetric else 0;
Puts.SetPaintingStrategy(PaintingStrategy.SQUARED_HISTOGRAM);
Puts.AssignValueColor(Color.DOWNTICK);

plot NetMetric = Calls + Puts;
NetMetric.AssignValueColor(Color.WHITE);
'''
    template = Template(template_code)
    thinkscript_code = template.render(call_codes=call_codes, put_codes=put_codes)
    return thinkscript_code


# # Fetch SPX price from Yahoo Finance
def tos_ts_0dte_spx(symbol='SPY'):
    spx = yf.Ticker("^SPX")
    spx_price = spx.history(period="1d").iloc[-1]['Close']
    rounded_spx_price = round(spx_price / 5) * 5

import datetime
import pandas as pd
import pandas_market_calendars as mcal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import functools

class MarketIntervalCalculator:
    """
    Computes the next update time for an application that polls during NYSE market hours.

    Intended to be instantiated once at application startup.
    Use get_next_update_time() for each web request.
    """

    NYSE_TIMEZONE_STR = 'America/New_York'

    def __init__(self):
        try:
            self._market_tz = ZoneInfo(self.NYSE_TIMEZONE_STR)
        except ZoneInfoNotFoundError as e:
            raise RuntimeError(f"Timezone not found: {e}")

        try:
            self._nyse_calendar = mcal.get_calendar('NYSE')
        except Exception as e:
            raise RuntimeError(f"Failed to load NYSE calendar: {e}")

    def get_market_close(self) -> datetime:
        now = datetime.datetime.now(self._market_tz)
        schedule_today = self._get_market_schedule_for_date(now.date())
        return schedule_today.iloc[0]['market_close'].astimezone(self._market_tz)

    @functools.lru_cache(maxsize=2)
    def _get_market_schedule_for_date(self, date_obj):
        """Returns NYSE open/close times for a given date."""
        return self._nyse_calendar.schedule(start_date=date_obj, end_date=date_obj)

    def get_next_update_time(self) -> datetime.datetime:
        """
        Returns the next datetime when the app should poll for updates:
        - If the market is open: next top-of-minute plus 5 seconds, capped at market close.
        - If the market is closed: next market open time.
        """
        now = datetime.datetime.now(self._market_tz)

        schedule_today = self._get_market_schedule_for_date(now.date())
        if schedule_today is not None and not schedule_today.empty:
            market_open = schedule_today.iloc[0]['market_open'].astimezone(self._market_tz)
            market_close = schedule_today.iloc[0]['market_close'].astimezone(self._market_tz)

            if market_open <= now < market_close:
                # Market is open — return next top-of-minute, capped at market_close
                next_minute = (now + datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)
                return min(next_minute, market_close)

            elif now < market_open:
                # Market not open yet today
                return market_open

        # Market closed today — look ahead for next open day
        check_date = now.date() + datetime.timedelta(days=1)
        for _ in range(10):  # Search up to 10 days ahead
            schedule = self._get_market_schedule_for_date(check_date)
            if schedule is not None and not schedule.empty:
                return schedule.iloc[0]['market_open'].astimezone(self._market_tz)
            check_date += datetime.timedelta(days=1)

        # If no open day found in 10 days, return 1 hour from now
        return now + datetime.timedelta(hours=1)

    def is_market_open(self) -> bool:
        """
        Checks if the NYSE market is currently open.

        Returns:
            bool: True if the market is open, False otherwise.
        """
        if self._market_tz is None or self._nyse_calendar is None:
            return False

        now = datetime.datetime.now(self._market_tz)
        schedule_today = self._get_market_schedule_for_date(now.date())

        if schedule_today is not None and not schedule_today.empty:
            market_open_today = schedule_today.iloc[0]['market_open'].astimezone(self._market_tz)
            market_close_today = schedule_today.iloc[0]['market_close'].astimezone(self._market_tz)
            return market_open_today <= now < market_close_today

        return False
