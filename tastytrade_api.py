import asyncio
from datetime import date, timedelta
from collections import defaultdict
from typing import Optional
from persist_cache import cache

from tastytrade.market_data import MarketData, a_get_market_data_by_type, get_market_data_by_type
from tastytrade.session import Session
from tastytrade.instruments import get_option_chain
from tastytrade.instruments import NestedOptionChain, Option

class TastyTradeAPI:
    def __init__(self, session: Session):
        self.session = session

    def __getstate__(self):
        # copy everything except things with lock such as Session
        state = self.__dict__.copy()
        state.pop("session", None)
        return state

    """
    def __setstate__(self, state):
        # restore data
        self.__dict__.update(state)
        # recreate a fresh lock
        self._lock = threading.RLock()
    # @cache(expiry=timedelta(hours=12))
    """

    @cache(expiry=timedelta(hours=12))
    def get_option_chain(self, symbol: str) -> dict[date, list[Option]]:
        """
        oc[today][0:1]
        """
        return get_option_chain(self.session, symbol)


    @cache(expiry=timedelta(hours=12))
    def get_nested_option_chain(self, symbol: str) -> NestedOptionChain:
        """
        noc.expirations[0]
        """
        return NestedOptionChain.get(self.session, symbol)[0]

    @cache(expiry=timedelta(hours=12))
    def get_0dte_option_symbols(self, symbol: str):
        noc = self.get_nested_option_chain(symbol)
        return [x for strike in noc.expirations[0].strikes for x in (strike.call, strike.put)]


    def get_market_data(self, kwargs):
        """
        Call the '/market-data/' end point to get real time trading data like last and volume.

        Example:
        quote = api.get_market_data({ 'equities':['SPY'] })[0]
        """

        # valid = {
        #     "cryptocurrencies",
        #     "equities",
        #     "futures",
        #     "future_options",
        #     "indices",
        #     "options",
        # }
        # if asset_type not in valid:
        #     raise ValueError(f"unknown asset type {asset_type!r}, must be one of {valid}")
        #return get_market_data_batch(self.session, **kwargs)
        return get_market_data_by_type(self.session, **kwargs)
    
    async def a_get_market_data_batch(self,
        cryptocurrencies: Optional[list[str]] = None,
        equities: Optional[list[str]] = None,
        futures: Optional[list[str]] = None,
        future_options: Optional[list[str]] = None,
        indices: Optional[list[str]] = None,
        options: Optional[list[str]] = None,
    ) -> list[MarketData]:
        session = self.session
        """
        Gets market data for the given symbols grouped by instrument type.
        This function will automatically handle chunking of symbols to respect the
        API limit of 100 symbols per request. This is the async version.

        :param session: active session to use
        :param cryptocurrencies: list of cryptocurrencies to fetch
        :param equities: list of equities to fetch
        :param futures: list of futures to fetch
        :param future_options: list of future options to fetch
        :param indices: list of indices to fetch
        :param options: list of options to fetch
        """
        # Create list[tuple[str,str]] = []
        all_symbols = [
            (sym_type, s)
            for sym_type, symbols in {
                "cryptocurrencies": cryptocurrencies,
                "equities": equities,
                "futures": futures,
                "future_options": future_options,
                "indices": indices,
                "options": options,
            }.items()
            if symbols
            for s in symbols
        ]

        tasks = []
        for i in range(0, len(all_symbols), 100):
            chunk = all_symbols[i : i + 100]
            kwargs: dict[str, list[str]] = defaultdict(list)
            for sym_type, symbol in chunk:
                kwargs[sym_type].append(symbol)
            tasks.append(a_get_market_data_by_type(session, **kwargs))

        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    def get_market_data_batch(self, kwargs) -> list[MarketData]:
        return get_market_data_batch(self.session, **kwargs)



def get_market_data_batch(
    session: Session,
    cryptocurrencies: Optional[list[str]] = None,
    equities: Optional[list[str]] = None,
    futures: Optional[list[str]] = None,
    future_options: Optional[list[str]] = None,
    indices: Optional[list[str]] = None,
    options: Optional[list[str]] = None,
) -> list[MarketData]:
    """
    Gets market data for the given symbols grouped by instrument type.
    This function will automatically handle chunking of symbols to respect the
    API limit of 100 symbols per request.

    """
    # Create list[tuple[str,str]] = []
    all_symbols = [
        (sym_type, s)
        for sym_type, symbols in {
            "cryptocurrencies": cryptocurrencies,
            "equities": equities,
            "futures": futures,
            "future_options": future_options,
            "indices": indices,
            "options": options,
        }.items()
        if symbols
        for s in symbols
    ]

    results = []
    for i in range(0, len(all_symbols), 100):
        chunk = all_symbols[i : i + 100]
        kwargs: dict[str, list[str]] = defaultdict(list)
        for sym_type, symbol in chunk:
            kwargs[sym_type].append(symbol)
        results.extend(get_market_data_by_type(session, **kwargs))

    return results


