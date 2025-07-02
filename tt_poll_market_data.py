import asyncio
import logging
import os
import sys
import pandas as pd
import pytz
import typer
from filelock import FileLock

from datetime import datetime, date
from decimal import Decimal
from typing import Annotated
from typer import Option

from tastytrade_api import TastyTradeAPI
from tastytrade.instruments import NestedOptionChain, Strike
from tastytrade.market_data import MarketData
from ttcli.utils import RenewableSession
from utils import MarketIntervalCalculator

# --- Logging Setup ---
logger = None
def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configures the application's logging.

    Args:
        verbose (bool): If True, sets logging level to DEBUG for detailed logs
                        in the file. Otherwise, sets to INFO.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Prevent adding multiple handlers if setup_logging is called more than once
    if not logger.handlers:
        # File Handler (all levels): Logs everything to a file
        file_handler = logging.FileHandler('poller.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console Handler (INFO and above): Only INFO, WARNING, ERROR, CRITICAL to console
        console_handler = logging.StreamHandler(sys.stdout) # Explicitly use stdout for console output
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger

def create_dataframe_from_market_data(
    market_data: list[MarketData],
    underlying_price: Decimal,
    created_at: datetime,
    expiration_date: date,
    sym2strike: dict[str,dict]
) -> pd.DataFrame:
    """Creates a Pandas DataFrame from a list of MarketData objects."""
    data = []
    for md in market_data:
        row = {
            'last_trade_at': pd.to_datetime(md.updated_at, utc=True),
            'created_at': created_at,
            'symbol': sym2strike[md.symbol]['streamer_symbol'],
            'putCall': sym2strike[md.symbol]['type'],
            'strike': sym2strike[md.symbol]['strike_price'],
            'bid': md.bid,
            'ask': md.ask,
            'price': md.last,
            'open_interest': md.open_interest,
            'delta': 0, #md.delta,
            'day_volume': md.volume,
            'underlying_price': underlying_price,
            'expiration_date': expiration_date,
        }
        data.append(row)
    df = pd.DataFrame(data)
    #logger.info(f"create_dataframe_from_market_data df.len={len(df)} max_last_trade_t={df.last_trade_at.max()}")
    return df

def merge_save_df(df: pd.DataFrame, df_new: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Merges two dataframes and saves the result to a parquet file.
    """
    if df_new is None or df_new.empty:
        return df

    if df is None:
        if os.path.exists(filename):
            df = pd.read_parquet(filename)
        else:
            logger.info(f"Creating new file: {filename}")
            df_new.to_parquet(filename)
            return df_new

    # Combine and remove duplicates, keeping the last observed data for each symbol
    # OLD if not df.empty and not df_new.empty and df.last_trade_at.max() >= df_new.last_trade_at.max():
    combined_df = pd.concat([df, df_new], ignore_index=True)
    combined_df.drop_duplicates(subset=['symbol', 'last_trade_at'], keep='last', inplace=True)

    # Only write to disk if there are actual changes
    if len(combined_df) > len(df):
        combined_df.to_parquet(filename)
        logger.info(f"Clobbered {filename} len={len(combined_df)}")
        return combined_df
    else:
        logger.info(f"No new trade data to save to {filename}")
        return df

async def poll_chain_data(
    symbol: str,
    strikes: int,
    filename: str,
):
    """
    Periodically polls for option chain data during market hours.
    """
    session = RenewableSession()
    api = TastyTradeAPI(session)
    market_interval_calculator = MarketIntervalCalculator()
    market_close_dt = market_interval_calculator.get_market_close()
    df = None
    sym2strike: dict[str,dict] = {}

    # Fetch option chain and filter strikes once
    logger.info("Fetching initial option chain and filtering strikes...")
    try:
        quote_list = api.get_market_data({'equities': [symbol]})
        if not quote_list:
            logger.error(f"Could not get initial quote for underlying {symbol}")
            return
        quote = quote_list[0]
        underlying_price = quote.last

        noc = api.get_nested_option_chain(symbol)
        if not noc.expirations:
            logger.error(f"No expirations found for {symbol}")
            return
        exp = noc.expirations[0]
        expiration_date = exp.expiration_date

        for s in exp.strikes:
            sym2strike[s.call] = s.model_dump()
            sym2strike[s.call]['type'] = 'CALL'
            sym2strike[s.call]['streamer_symbol'] = s.call_streamer_symbol
            sym2strike[s.put] = s.model_dump()
            sym2strike[s.put]['type'] = 'PUT'
            sym2strike[s.put]['streamer_symbol'] = s.put_streamer_symbol

        all_strikes = sorted(exp.strikes, key=lambda s: s.strike_price)
        mid_index = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i].strike_price - underlying_price))

        half = strikes // 2
        start = max(0, mid_index - half)
        end = min(len(all_strikes), mid_index + half)
        selected_strikes = all_strikes[start:end]

        option_symbols = [s.call for s in selected_strikes] + [s.put for s in selected_strikes]
        logger.info(f"Monitoring {len(option_symbols)} option symbols.")

    except Exception as e:
        logger.error(f"Error during initial setup: {e}", exc_info=True)
        return

    while True: #market_interval_calculator.is_market_open():
        now = datetime.now(market_close_dt.tzinfo)
        if now >= market_close_dt:
            break
        try:
            # 1. Get underlying quote
            quote_list = api.get_market_data({'equities': [symbol]})
            if not quote_list:
                logger.warning(f"Could not get quote for underlying {symbol}")
                await asyncio.sleep(60)
                continue
            quote = quote_list[0]
            underlying_price = quote.last

            # 2. Get market data for the pre-selected options
            md = await api.a_get_market_data_batch(options=option_symbols)

            # 3. Create and save dataframe
            created_at = datetime.now(pytz.utc)
            df_new = create_dataframe_from_market_data(md, underlying_price, created_at, expiration_date, sym2strike)
            df = merge_save_df(df, df_new, filename)
            #logger.info(f"Fetched and saved data for {len(df_new)} options to {filename}")

        except Exception as e:
            logger.error(f"Error polling market data: {e}", exc_info=True)

        next_update_time = market_interval_calculator.get_next_update_time()
        now_in_market_tz = datetime.now(next_update_time.tzinfo)
        next_interval = (next_update_time - now_in_market_tz).total_seconds()
        next_interval = max(1, next_interval)

        logger.info(f"Sleeping for {int(next_interval)} seconds until {next_update_time}.")
        await asyncio.sleep(next_interval)

    logger.info("Market is closed. Exiting.")


cli = typer.Typer(no_args_is_help=True)

@cli.command(help="Polls 0DTE option chain for a given symbol and saves it to a parquet file.")
def main(
    symbol: Annotated[str, Option(help="The underlying symbol to poll for (e.g., SPY).")] = 'SPY',
    strikes: Annotated[int, Option("--strikes", "-s", help="The number of strikes to fetch around the money.")] = 40,
    filename: Annotated[str, Option("--file", help="Filename for the dataframe.")] = None,
    verbose: Annotated[bool, Option("--verbose", "-v", help="Enable verbose logging.")] = False,
):
    """
    Main entry point for the polling application.
    """
    global logger
    logger = setup_logging(verbose)

    if filename is None:
        ymd = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
        filename = f'{symbol}.{ymd}.chain.parquet'

    lock_file_path = f".{symbol}.lock"
    lock = FileLock(lock_file_path)

    try:
        with lock:
            logger.info(f"Acquired lock for {symbol}.")
            # Check if market is open before starting
            market_interval_calculator = MarketIntervalCalculator()
            if not market_interval_calculator.is_market_open():
                logger.info("Market is currently closed. Exiting.")
                sys.exit(0)

            asyncio.run(poll_chain_data(symbol, strikes, filename))
    except TimeoutError:
        logger.warning(f"Another instance of the poller for {symbol} is already running. Exiting.")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Polling stopped by user.")
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()
