"""Market data layer backed by the Binance public data archive.

The original PGPortfolio pulled 30-minute candles from the (long dead)
Poloniex public API.  This module replaces that data source with
https://data.binance.vision - Binance's public S3 archive of historical
klines.  The archive keeps data of *delisted* symbols as well, which lets the
dynamic universe selection remain free of survivorship bias.

Data is cached locally in an sqlite database (``./database/dynamic_data.db``)
so repeated runs do not re-download anything.
"""

from __future__ import absolute_import, division, print_function

import io
import logging
import re
import sqlite3
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

ARCHIVE_HOST = "https://data.binance.vision"
LIST_HOST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DATABASE_PATH = "./database/dynamic_data.db"

KLINE_COLUMNS = ["ts", "open", "high", "low", "close", "volume",
                 "quote_volume", "trades"]


def month_range(start, end):
    """list of 'YYYY-MM' strings from start to end (inclusive).

    :param start/end: anything pandas can parse into a timestamp.
    """
    periods = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    return [str(p) for p in periods]


class BinanceArchive(object):
    def __init__(self, db_path=DATABASE_PATH, max_workers=16):
        self._db_path = db_path
        self._max_workers = max_workers
        self._local = threading.local()
        self._db_lock = threading.Lock()
        self._initialize_db()

    # ------------------------------------------------------------------ http

    @property
    def _session(self):
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def _get(self, url, retries=5):
        delay = 1.0
        for attempt in range(retries):
            try:
                response = self._session.get(url, timeout=60)
                if response.status_code == 200:
                    return response.content
                if response.status_code == 404:
                    return None
                logging.warning("GET %s -> %s", url, response.status_code)
            except requests.RequestException as e:
                logging.warning("GET %s failed: %s", url, e)
            time.sleep(delay)
            delay *= 2
        raise IOError("could not fetch %s after %d attempts" % (url, retries))

    # ---------------------------------------------------------------- sqlite

    def _connect(self):
        connection = sqlite3.connect(self._db_path, timeout=120)
        return connection

    def _initialize_db(self):
        with self._connect() as connection:
            cursor = connection.cursor()
            for table in ("daily", "klines30m", "klines_perp4h"):
                cursor.execute(
                    "CREATE TABLE IF NOT EXISTS {} (symbol VARCHAR(20), ts INTEGER,"
                    " open FLOAT, high FLOAT, low FLOAT, close FLOAT,"
                    " volume FLOAT, quote_volume FLOAT, trades INTEGER,"
                    " taker_buy FLOAT, PRIMARY KEY (symbol, ts));".format(table))
                try:  # migrate older caches in place
                    cursor.execute(
                        "ALTER TABLE {} ADD COLUMN taker_buy FLOAT;".format(table))
                except sqlite3.OperationalError:
                    pass
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS funding (symbol VARCHAR(20),"
                " ts INTEGER, rate FLOAT, PRIMARY KEY (symbol, ts));")
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS downloaded_months (symbol VARCHAR(20),"
                " interval VARCHAR(8), month VARCHAR(8), rows INTEGER,"
                " PRIMARY KEY (symbol, interval, month));")
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS symbol_months (symbol VARCHAR(20),"
                " interval VARCHAR(8), months TEXT, fetched_at INTEGER,"
                " PRIMARY KEY (symbol, interval));")
            connection.commit()

    # -------------------------------------------------------------- listings

    def list_symbols(self, quote=None):
        """All symbols that ever had spot monthly klines (including delisted)."""
        symbols = []
        marker = ""
        prefix = "data/spot/monthly/klines/"
        while True:
            url = "{}?delimiter=/&prefix={}{}".format(
                LIST_HOST, prefix, "&marker=" + marker if marker else "")
            body = self._get(url).decode()
            symbols += [p.split("/")[-2] for p in
                        re.findall(r"<Prefix>([^<]+)</Prefix>", body)
                        if p != prefix]
            truncated = "<IsTruncated>true</IsTruncated>" in body
            if not truncated:
                break
            next_marker = re.search(r"<NextMarker>([^<]+)</NextMarker>", body)
            marker = next_marker.group(1)
        if quote:
            symbols = [s for s in symbols if s.endswith(quote)]
        return symbols

    @staticmethod
    def _month_prefix(symbol, interval):
        if interval == "fundingRate":
            return "data/futures/um/monthly/fundingRate/{}/".format(symbol)
        if interval == "perp4h":
            return "data/futures/um/monthly/klines/{}/4h/".format(symbol)
        return "data/spot/monthly/klines/{}/{}/".format(symbol, interval)

    @staticmethod
    def _table_for(interval):
        return {"1d": "daily", "30m": "klines30m",
                "perp4h": "klines_perp4h"}[interval]

    def list_symbol_months(self, symbol, interval):
        """Which monthly zip files exist for a symbol/interval (cached)."""
        with self._db_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT months, fetched_at FROM symbol_months WHERE symbol=? AND interval=?;",
                (symbol, interval)).fetchone()
        now = int(time.time())
        # refresh the cached listing once a week so new months show up
        if row is not None and now - row[1] < 7 * 24 * 3600:
            return row[0].split(",") if row[0] else []
        prefix = self._month_prefix(symbol, interval)
        body = self._get("{}?delimiter=/&prefix={}".format(LIST_HOST, prefix))
        months = []
        if body is not None:
            keys = re.findall(r"<Key>([^<]+\.zip)</Key>", body.decode())
            months = sorted(re.search(r"-(\d{4}-\d{2})\.zip$", k).group(1)
                            for k in keys)
        with self._db_lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO symbol_months VALUES (?,?,?,?);",
                (symbol, interval, ",".join(months), now))
            connection.commit()
        return months

    # -------------------------------------------------------------- download

    @staticmethod
    def _parse_kline_zip(content):
        """Parse a monthly kline zip into a list of KLINE_COLUMNS tuples."""
        archive = zipfile.ZipFile(io.BytesIO(content))
        raw = archive.read(archive.namelist()[0]).decode()
        rows = []
        for line in raw.splitlines():
            if not line or line.startswith("open_time"):
                continue
            parts = line.split(",")
            ts = int(parts[0])
            if ts > 10 ** 14:      # microseconds (archive format since 2025)
                ts //= 10 ** 6
            else:                  # milliseconds
                ts //= 10 ** 3
            rows.append((ts, float(parts[1]), float(parts[2]), float(parts[3]),
                         float(parts[4]), float(parts[5]), float(parts[7]),
                         int(parts[8]), float(parts[9])))
        return rows

    @staticmethod
    def _parse_funding_zip(content):
        archive = zipfile.ZipFile(io.BytesIO(content))
        raw = archive.read(archive.namelist()[0]).decode()
        rows = []
        for line in raw.splitlines():
            if not line or line.startswith("calc_time"):
                continue
            parts = line.split(",")
            ts = int(parts[0])
            ts //= 10 ** 6 if ts > 10 ** 14 else 10 ** 3
            rows.append((ts, float(parts[2])))
        return rows

    def _download_month(self, symbol, interval, month):
        if interval == "fundingRate":
            url = "{}/{}{}-fundingRate-{}.zip".format(
                ARCHIVE_HOST, self._month_prefix(symbol, interval), symbol, month)
            content = self._get(url)
            rows = self._parse_funding_zip(content) if content is not None else []
            with self._db_lock, self._connect() as connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO funding VALUES (?,?,?);",
                    [(symbol,) + row for row in rows])
                connection.execute(
                    "INSERT OR REPLACE INTO downloaded_months VALUES (?,?,?,?);",
                    (symbol, interval, month, len(rows)))
                connection.commit()
            return len(rows)
        suffix = "4h" if interval == "perp4h" else interval
        url = "{}/{}{}-{}-{}.zip".format(
            ARCHIVE_HOST, self._month_prefix(symbol, interval), symbol, suffix, month)
        content = self._get(url)
        rows = self._parse_kline_zip(content) if content is not None else []
        table = self._table_for(interval)
        with self._db_lock, self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO {} VALUES (?,?,?,?,?,?,?,?,?,?);".format(table),
                [(symbol,) + row for row in rows])
            connection.execute(
                "INSERT OR REPLACE INTO downloaded_months VALUES (?,?,?,?);",
                (symbol, interval, month, len(rows)))
            connection.commit()
        return len(rows)

    def read_funding(self, symbol, start_ts=None, end_ts=None):
        query = "SELECT ts, rate FROM funding WHERE symbol=?"
        args = [symbol]
        if start_ts is not None:
            query += " AND ts>=?"
            args.append(int(start_ts))
        if end_ts is not None:
            query += " AND ts<?"
            args.append(int(end_ts))
        with self._connect() as connection:
            return pd.read_sql_query(query + " ORDER BY ts;", connection,
                                     params=args, index_col="ts")

    def ensure_data(self, symbols, interval, start, end, log_every=200):
        """Make sure candles for symbols between start and end are cached.

        Months that do not exist in the archive (before listing / after
        delisting) are skipped based on the archive's own file listing.
        """
        wanted_months = set(month_range(start, end))
        with self._connect() as connection:
            have = {row[0] for row in connection.execute(
                "SELECT symbol || '|' || month FROM downloaded_months"
                " WHERE interval=?;", (interval,)).fetchall()}

        def month_jobs(symbol):
            available = self.list_symbol_months(symbol, interval)
            return [(symbol, month) for month in available
                    if month in wanted_months and
                    "{}|{}".format(symbol, month) not in have]

        jobs = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            for job_list in executor.map(month_jobs, symbols):
                jobs.extend(job_list)
        logging.info("downloading %d monthly %s files for %d symbols",
                     len(jobs), interval, len(symbols))
        done = 0
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [executor.submit(self._download_month, s, interval, m)
                       for s, m in jobs]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % log_every == 0:
                    logging.info("downloaded %d/%d files", done, len(jobs))
        return len(jobs)

    # ------------------------------------------------------------------ read

    def read_frame(self, symbol, interval, start_ts=None, end_ts=None):
        """Read cached candles as a DataFrame indexed by open-time (seconds)."""
        table = self._table_for(interval)
        query = "SELECT ts, open, high, low, close, volume, quote_volume," \
                " trades, taker_buy FROM {} WHERE symbol=?".format(table)
        args = [symbol]
        if start_ts is not None:
            query += " AND ts>=?"
            args.append(int(start_ts))
        if end_ts is not None:
            query += " AND ts<?"
            args.append(int(end_ts))
        with self._connect() as connection:
            frame = pd.read_sql_query(query + " ORDER BY ts;", connection,
                                      params=args, index_col="ts")
        return frame

    def read_daily_panel(self, symbols, start_ts, end_ts, column="quote_volume"):
        """dict of symbol -> daily DataFrame, token-swap sanitized."""
        result = {}
        for symbol in symbols:
            frame = self.read_frame(symbol, "1d", start_ts, end_ts)
            if len(frame):
                result[symbol] = sanitize_token_swaps(frame, gap_days=3, jump=5.0)
        return result


def sanitize_token_swaps(frame, gap_days=3, jump=5.0):
    """Truncate a candle series at token-swap style discontinuities.

    Some archive symbols were re-used for a different token after a
    redenomination (e.g. LUNAUSDT: Terra Classic LUNA until May 2022, the
    new-chain LUNA afterwards).  Those show up as a multi-day gap in the data
    combined with a huge price jump.  Trading across such a gap would credit
    the backtest with a fictional (often 1000x) return, so the series is cut
    at the first such discontinuity and the rest is discarded.
    """
    if len(frame) < 2:
        return frame
    index = frame.index.to_numpy()
    close = frame["close"].to_numpy()
    open_ = frame["open"].to_numpy()
    gaps = index[1:] - index[:-1]
    ratio = open_[1:] / close[:-1]
    bad = (gaps > gap_days * 86400) & ((ratio > jump) | (ratio < 1.0 / jump))
    if bad.any():
        cut = bad.argmax() + 1
        frame = frame.iloc[:cut]
    return frame
