"""
Data Store Module
================

Manages data persistence and caching:
- Local database storage
- Incremental data updates
- Cache management
- Data versioning
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import json
import pickle
import sqlite3
import os

import pandas as pd
import numpy as np

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"project_root: {project_root}")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

logger = logging.getLogger(__name__)


class DataStore:
    """Data store for managing persistent data storage."""

    def __init__(self, base_dir: str = None):
        """
        Initialize data store.

        Args:
            base_dir: Base directory for data storage (uses config.data.base_dir if None)
        """
        # Use config settings if base_dir not provided
        if base_dir is None:
            try:
                from src.config.settings import get_config
                config = get_config()
                base_dir = config.data.base_dir
            except Exception as e:
                logger.warning(f"Failed to load config, using default './data': {e}")
                base_dir = './data'
        
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "processed"
        self.db_path = self.base_dir / "finrl_trading.db"

        # Create directories
        for dir_path in [self.base_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create price data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_price_ticker_date 
                ON price_data(ticker, date)
            ''')

            # Create fundamental data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fundamental_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    gvkey TEXT,
                    gsector TEXT,
                    prccd REAL,
                    ajexdi REAL,
                    adj_close REAL,
                    adj_close_q REAL,
                    eps REAL,
                    bps REAL,
                    dps REAL,
                    pe REAL,
                    pb REAL,
                    ps REAL,
                    roe REAL,
                    cur_ratio REAL,
                    quick_ratio REAL,
                    cash_ratio REAL,
                    acc_rec_turnover REAL,
                    debt_ratio REAL,
                    debt_to_equity REAL,
                    net_income_ratio REAL,
                    market_cap REAL,
                    y_return REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_fundamental_ticker_date 
                ON fundamental_data(ticker, date)
            ''')

            # Create S&P 500 components table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sp500_components (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    tickers TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            ''')

            # Create legacy tables for backward compatibility
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT,
                    metadata TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    cache_key TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    file_path TEXT,
                    metadata TEXT
                )
            ''')

            # New: table for versioned binary objects (DataFrames, etc.)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_blob BLOB,
                    metadata TEXT,
                    UNIQUE(data_type, version)
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_data_objects_type_created
                ON data_objects(data_type, created_at DESC)
            ''')

            # New: table for cache entries stored directly in the database
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    data_blob BLOB,
                    metadata TEXT
                )
            ''')

            conn.commit()
            logger.info(f"Initialized database at {self.db_path}")

    def save_dataframe(self, df: pd.DataFrame, name: str,
                      version: str = None, metadata: Dict = None) -> str:
        """
        Save DataFrame to database with versioning (binary BLOB).

        Args:
            df: DataFrame to save
            name: Name identifier for the data
            version: Version string (auto-generated if None)
            metadata: Additional metadata

        Returns:
            Version string of the saved object
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Serialize DataFrame to binary (pickle)
        try:
            data_blob = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to serialize DataFrame for {name}@{version}: {e}")
            raise

        # Insert/replace into data_objects
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO data_objects (data_type, version, data_blob, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(data_type, version) DO UPDATE SET
                    data_blob=excluded.data_blob,
                    metadata=excluded.metadata
            ''', (
                name,
                version,
                data_blob,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()

            # Also record to legacy data_versions table for bookkeeping (file_path NULL)
            try:
                cursor.execute('''
                    INSERT INTO data_versions (data_type, version, file_path, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (name, version, None, json.dumps(metadata) if metadata else None))
                conn.commit()
            except Exception:
                # ignore if legacy table constraints differ
                pass

        logger.info(f"Saved DataFrame '{name}' version {version} to database")
        return version

    def load_dataframe(self, name: str, version: str = None) -> Optional[pd.DataFrame]:
        """
        Load versioned DataFrame from database.

        Args:
            name: Name identifier for the data
            version: Specific version to load (latest if None)

        Returns:
            Loaded DataFrame or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if version:
                cursor.execute('''
                    SELECT data_blob FROM data_objects
                    WHERE data_type = ? AND version = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (name, version))
            else:
                cursor.execute('''
                    SELECT data_blob FROM data_objects
                    WHERE data_type = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (name,))
            row = cursor.fetchone()

        if row and row[0] is not None:
            try:
                return pickle.loads(row[0])
            except Exception as e:
                logger.error(f"Failed to deserialize DataFrame '{name}' version '{version or 'latest'}': {e}")
                return None

        # Backward compatibility: try file-based loading if legacy record exists
        file_path = self._get_file_path(name, version)
        if file_path and file_path.exists():
            logger.info(f"Loading legacy DataFrame from {file_path}")
            return pd.read_csv(file_path)
        return None

    def _save_version_info(self, data_type: str, version: str,
                          file_path: str, metadata: Dict = None):
        """Save version information to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO data_versions (data_type, version, file_path, metadata)
                VALUES (?, ?, ?, ?)
            ''', (data_type, version, file_path, json.dumps(metadata) if metadata else None))
            conn.commit()

    def _get_file_path(self, name: str, version: str = None) -> Optional[Path]:
        """Get file path for given name and version."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if version:
                cursor.execute('''
                    SELECT file_path FROM data_versions
                    WHERE data_type = ? AND version = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (name, version))
            else:
                cursor.execute('''
                    SELECT file_path FROM data_versions
                    WHERE data_type = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (name,))

            result = cursor.fetchone()
            return Path(result[0]) if result else None

    def list_versions(self, data_type: str) -> List[Dict]:
        """List all versions for a data type."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT version, created_at, metadata FROM data_versions
                WHERE data_type = ?
                ORDER BY created_at DESC
            ''', (data_type,))

            versions = []
            for row in cursor.fetchall():
                versions.append({
                    'version': row[0],
                    'created_at': row[1],
                    'metadata': json.loads(row[2]) if row[2] else None
                })

            return versions

    def cache_data(self, key: str, data: Any, ttl_hours: int = 24) -> str:
        """
        Cache data with time-to-live (stored in SQLite).

        Args:
            key: Cache key
            data: Data to cache (DataFrame, dict, etc.)
            ttl_hours: Time-to-live in hours

        Returns:
            Cache key
        """
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        # Serialize data
        try:
            if isinstance(data, pd.DataFrame):
                blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                metadata = json.dumps({'type': 'dataframe'})
            elif isinstance(data, dict):
                blob = json.dumps(data).encode('utf-8')
                metadata = json.dumps({'type': 'json'})
            else:
                blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                metadata = json.dumps({'type': 'pickle'})
        except Exception as e:
            logger.error(f"Failed to serialize cache data for key '{key}': {e}")
            raise

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO cache_entries (cache_key, expires_at, data_blob, metadata)
                VALUES (?, ?, ?, ?)
            ''', (key, expires_at.isoformat(), blob, metadata))
            conn.commit()

        logger.info(f"Cached data with key '{key}' to SQLite cache_entries")
        return key

    def get_cached_data(self, key: str) -> Optional[Any]:
        """
        Retrieve cached data if not expired (from SQLite cache_entries or legacy file cache).

        Args:
            key: Cache key

        Returns:
            Cached data or None if expired/not found
        """
        # First try SQLite cache_entries
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT data_blob, expires_at, metadata FROM cache_entries
                WHERE cache_key = ?
            ''', (key,))
            row = cursor.fetchone()
            if row:
                data_blob, expires_at, metadata = row
                if expires_at and datetime.now() > datetime.fromisoformat(expires_at):
                    logger.info(f"Cache expired for key '{key}' (cache_entries)")
                    return None
                try:
                    meta = json.loads(metadata) if metadata else {}
                except Exception:
                    meta = {}
                try:
                    if meta.get('type') == 'json':
                        return json.loads(data_blob.decode('utf-8'))
                    # default: pickle
                    return pickle.loads(data_blob)
                except Exception as e:
                    logger.warning(f"Failed to deserialize cache_entries for key '{key}': {e}")
                    return None

        # Legacy fallback: cache_metadata pointing to files
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT file_path, expires_at FROM cache_metadata
                WHERE cache_key = ?
            ''', (key,))
            result = cursor.fetchone()
            if not result:
                return None
            file_path, expires_at = result
            if expires_at and datetime.now() > datetime.fromisoformat(expires_at):
                logger.info(f"Cache expired for key '{key}' (legacy)")
                return None
            file_path = Path(file_path)
            if not file_path.exists():
                return None
            if file_path.suffix == '.pkl':
                return pd.read_pickle(file_path)
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            return pd.read_csv(file_path)

    def cleanup_expired_cache(self):
        """Clean up expired cache entries (SQLite and legacy file-based)."""
        now_iso = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Clean SQLite cache_entries
            cursor.execute('''
                DELETE FROM cache_entries
                WHERE expires_at IS NOT NULL AND expires_at < ?
            ''', (now_iso,))
            deleted_sqlite = cursor.rowcount or 0

            # Legacy: remove expired files and metadata
            cursor.execute('''
                SELECT cache_key, file_path FROM cache_metadata
                WHERE expires_at < ?
            ''', (now_iso,))
            expired_entries = cursor.fetchall()
            for key, file_path in expired_entries:
                try:
                    Path(file_path).unlink(missing_ok=True)
                    logger.info(f"Deleted expired cache file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {file_path}: {e}")
            cursor.execute('''
                DELETE FROM cache_metadata
                WHERE expires_at < ?
            ''', (now_iso,))

            conn.commit()

        logger.info(f"Cleaned up {deleted_sqlite} expired SQLite cache entries and {len(expired_entries)} legacy entries")

    def save_price_data(self, df: pd.DataFrame) -> int:
        """
        Save price data to database (upsert).
        
        Args:
            df: DataFrame with columns: ticker, datadate, prcod, prchd, prcld, prccd, adj_close, cshtrd
            
        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            return 0
        
        # Standardize column names
        df = df.copy()
        column_mapping = {
            'tic': 'ticker',
            'datadate': 'date',
            'prcod': 'open',
            'prchd': 'high',
            'prcld': 'low',
            'prccd': 'close',
            'cshtrd': 'volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'ticker':
                    df['ticker'] = df.get('gvkey', 'UNKNOWN')
                elif col == 'date':
                    df['date'] = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('datadate')
                else:
                    df[col] = np.nan
        
        # Convert date to string format
        if not isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        rows_affected = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO price_data 
                        (ticker, date, open, high, low, close, adj_close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['ticker'],
                        row['date'],
                        float(row['open']) if pd.notna(row['open']) else None,
                        float(row['high']) if pd.notna(row['high']) else None,
                        float(row['low']) if pd.notna(row['low']) else None,
                        float(row['close']) if pd.notna(row['close']) else None,
                        float(row['adj_close']) if pd.notna(row['adj_close']) else None,
                        float(row['volume']) if pd.notna(row['volume']) else None
                    ))
                    rows_affected += 1
                except Exception as e:
                    logger.warning(f"Failed to save price data for {row.get('ticker')} on {row.get('date')}: {e}")
                    continue
            
            conn.commit()
        
        logger.info(f"Saved {rows_affected} price data records to database")
        return rows_affected

    def get_price_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get price data from database.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with price data
        """
        if not tickers:
            return pd.DataFrame()
        
        placeholders = ','.join(['?' for _ in tickers])
        query = f'''
            SELECT ticker, date, open, high, low, close, adj_close, volume
            FROM price_data
            WHERE ticker IN ({placeholders})
            AND date >= ? AND date <= ?
            ORDER BY ticker, date
        '''
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=tickers + [start_date, end_date])
        
        if not df.empty:
            # Rename to match expected format
            df = df.rename(columns={
                'ticker': 'tic',
                'date': 'datadate',
                'open': 'prcod',
                'high': 'prchd',
                'low': 'prcld',
                'close': 'prccd',
                'volume': 'cshtrd'
            })
            df['gvkey'] = df['tic']
            
        return df

    def get_missing_price_dates(self, ticker: str, start_date: str, end_date: str, exchange: str = 'NYSE') -> List[Tuple[str, str]]:
        """
        Identify missing date ranges for price data using real trading calendar.
        
        Args:
            ticker: Ticker symbol
            start_date: Requested start date (YYYY-MM-DD)
            end_date: Requested end date (YYYY-MM-DD)
            exchange: Exchange name (default: NYSE)
        Returns:
            List of (start_date, end_date) tuples for missing ranges
            
        Note:
            - Uses real NYSE trading calendar to determine trading days
            - Only reports missing trading days, excludes weekends and holidays
            - If trading calendar library is not available, falls back to business days
        """
        from src.data.trading_calendar import get_missing_trading_days, consolidate_date_ranges
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all existing dates for this ticker in the range
            cursor.execute('''
                SELECT date
                FROM price_data
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date
            ''', (ticker, start_date, end_date))
            
            existing_dates = [row[0] for row in cursor.fetchall()]
            
            if not existing_dates:
                # No data exists for this ticker in the range
                return [(start_date, end_date)]
            
            # Use trading calendar to find missing trading days
            missing_trading_days = get_missing_trading_days(
                existing_dates, 
                start_date, 
                end_date,
                exchange=exchange
            )
            
            if not missing_trading_days:
                # No missing trading days
                return []
            
            # Consolidate consecutive dates into ranges
            missing_ranges = consolidate_date_ranges(missing_trading_days)
        
        return missing_ranges

    def save_fundamental_data(self, df: pd.DataFrame) -> int:
        """
        Save fundamental data to database (upsert).
        
        Args:
            df: DataFrame with fundamental data
            
        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            return 0
        
        df = df.copy()
        
        # Standardize column names
        if 'tic' in df.columns and 'ticker' not in df.columns:
            df['ticker'] = df['tic']
        if 'datadate' in df.columns and 'date' not in df.columns:
            df['date'] = df['datadate']
        
        # Convert date to string format
        if 'date' in df.columns and not isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        rows_affected = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    # Build dynamic column list based on available columns
                    columns = ['ticker', 'date']
                    values = [row.get('ticker', row.get('tic')), row['date']]
                    
                    optional_fields = [
                        'gvkey', 'gsector', 'prccd', 'ajexdi', 'adj_close', 'adj_close_q',
                        'eps', 'bps', 'dps', 'pe', 'pb', 'ps', 'roe',
                        'cur_ratio', 'quick_ratio', 'cash_ratio', 'acc_rec_turnover',
                        'debt_ratio', 'debt_to_equity', 'net_income_ratio', 'market_cap', 'y_return'
                    ]
                    
                    for field in optional_fields:
                        # Handle different column name conventions
                        field_value = None
                        if field in row:
                            field_value = row[field]
                        elif field.upper() in row:
                            field_value = row[field.upper()]
                        elif field == 'eps' and 'EPS' in row:
                            field_value = row['EPS']
                        elif field == 'bps' and 'BPS' in row:
                            field_value = row['BPS']
                        elif field == 'dps' and 'DPS' in row:
                            field_value = row['DPS']
                        
                        if field_value is not None and pd.notna(field_value):
                            columns.append(field)
                            values.append(float(field_value) if isinstance(field_value, (int, float, np.number)) else str(field_value))
                    
                    placeholders = ','.join(['?' for _ in columns])
                    update_clause = ','.join([f"{col}=excluded.{col}" for col in columns if col not in ['ticker', 'date']])
                    
                    cursor.execute(f'''
                        INSERT INTO fundamental_data ({','.join(columns)})
                        VALUES ({placeholders})
                        ON CONFLICT(ticker, date) DO UPDATE SET {update_clause}
                    ''', values)
                    rows_affected += 1
                except Exception as e:
                    logger.warning(f"Failed to save fundamental data for {row.get('ticker', row.get('tic'))} on {row.get('date')}: {e}")
                    continue
            
            conn.commit()
        
        logger.info(f"Saved {rows_affected} fundamental data records to database")
        return rows_affected

    # def get_fundamental_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    #     """
    #     Get fundamental data from database.
        
    #     Args:
    #         tickers: List of ticker symbols
    #         start_date: Start date (YYYY-MM-DD)
    #         end_date: End date (YYYY-MM-DD)
            
    #     Returns:
    #         DataFrame with fundamental data
    #     """
    #     if not tickers:
    #         return pd.DataFrame()
        
    #     placeholders = ','.join(['?' for _ in tickers])
    #     query = f'''
    #         SELECT *
    #         FROM fundamental_data
    #         WHERE ticker IN ({placeholders})
    #         AND date >= ? AND date <= ?
    #         ORDER BY ticker, date
    #     '''
        
    #     with sqlite3.connect(self.db_path) as conn:
    #         df = pd.read_sql_query(query, conn, params=tickers + [start_date, end_date])
        
    #     if not df.empty:
    #         # Remove internal columns and rename to match expected format
    #         df = df.drop(columns=['id', 'created_at'], errors='ignore')
    #         if 'ticker' in df.columns:
    #             df['tic'] = df['ticker']
    #         if 'date' in df.columns:
    #             df['datadate'] = df['date']
            
    #     return df

    def get_missing_fundamental_dates(self, ticker: str, start_date: str, end_date: str) -> List[Tuple[str, str]]:
        """
        Identify missing date ranges for fundamental data (quarterly).
        
        Args:
            ticker: Ticker symbol
            start_date: Requested start date (YYYY-MM-DD)
            end_date: Requested end date (YYYY-MM-DD)
            
        Returns:
            List of (start_date, end_date) tuples for missing ranges
            
        Note:
            - Checks for gaps in the entire date range, not just endpoints
            - For quarterly data, gaps > 100 days likely indicate missing quarters
            - Typical quarter = ~90 days, so 100-day threshold is reasonable
            - Reports ranges that likely contain missing quarterly reports
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all existing dates for this ticker in the range
            cursor.execute('''
                SELECT date
                FROM fundamental_data
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date
            ''', (ticker, start_date, end_date))
            
            existing_dates = [row[0] for row in cursor.fetchall()]
            
            if not existing_dates:
                # No data exists for this ticker in the range
                return [(start_date, end_date)]
            
            # Convert to datetime objects
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            existing_dts = pd.to_datetime(existing_dates)
            
            missing_ranges = []
            
            # For quarterly data, we use a 100-day threshold
            # This is reasonable since a quarter is ~90 days
            QUARTER_THRESHOLD_DAYS = 100
            
            # Check if we need data before the first existing date
            first_existing = existing_dts[0]
            if start_dt < first_existing:
                gap_days = (first_existing - start_dt).days
                if gap_days > QUARTER_THRESHOLD_DAYS:
                    range_end = (first_existing - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    missing_ranges.append((start_date, range_end))
            
            # Check for gaps in the middle (missing quarters)
            for i in range(len(existing_dts) - 1):
                current_date = existing_dts[i]
                next_date = existing_dts[i + 1]
                gap_days = (next_date - current_date).days
                
                # If gap > 100 days, likely missing one or more quarters
                if gap_days > QUARTER_THRESHOLD_DAYS:
                    gap_start = (current_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    gap_end = (next_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    missing_ranges.append((gap_start, gap_end))
            
            # Check if we need data after the last existing date
            last_existing = existing_dts[-1]
            if end_dt > last_existing:
                gap_days = (end_dt - last_existing).days
                if gap_days > QUARTER_THRESHOLD_DAYS:
                    range_start = (last_existing + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    missing_ranges.append((range_start, end_date))
        
        return missing_ranges

    def save_sp500_components(self, date: str, tickers: str) -> bool:
        """
        Save S&P 500 components to database.
        
        Args:
            date: Date for the components (YYYY-MM-DD)
            tickers: Comma-separated ticker string
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO sp500_components (date, tickers)
                    VALUES (?, ?)
                ''', (date, tickers))
                conn.commit()
            
            logger.info(f"Saved S&P 500 components for {date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save S&P 500 components: {e}")
            return False

    def get_sp500_components(self, date: str = None) -> Optional[str]:
        """
        Get S&P 500 components from database.
        
        Args:
            date: Date for the components (latest if None)
            
        Returns:
            Comma-separated ticker string or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if date:
                cursor.execute('''
                    SELECT tickers FROM sp500_components
                    WHERE date = ?
                ''', (date,))
            else:
                cursor.execute('''
                    SELECT tickers FROM sp500_components
                    ORDER BY date DESC LIMIT 1
                ''')
            
            result = cursor.fetchone()
            return result[0] if result else None

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        total_size = 0
        file_count = 0

        for dir_path in [self.processed_dir]:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

        # Add database size
        if self.db_path.exists():
            total_size += self.db_path.stat().st_size

        # Database stats
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM data_versions")
            version_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM price_data")
            price_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM fundamental_data")
            fundamental_count = cursor.fetchone()[0]

            # New tables
            try:
                cursor.execute("SELECT COUNT(*) FROM data_objects")
                objects_count = cursor.fetchone()[0]
            except Exception:
                objects_count = 0
            try:
                cursor.execute("SELECT COUNT(*) FROM cache_entries")
                cache_entries_count = cursor.fetchone()[0]
            except Exception:
                cache_entries_count = 0

        return {
            'total_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'data_versions': version_count,
            'price_records': price_count,
            'fundamental_records': fundamental_count,
            'data_objects': objects_count,
            'cache_entries': cache_entries_count,
            'database_path': str(self.db_path)
        }

    # =========================
    # Raw fundamentals helpers
    # =========================
    def _save_raw_payload(self, source: str, ticker: Optional[str], payload: str,
                           start_date: str, end_date: str, data: Any,
                           extra_meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Save raw fundamentals payload into data_objects table.

        Args:
            source: 'FMP' | 'Yahoo' | 'WRDS' etc.
            ticker: Ticker symbol or None for bulk
            payload: e.g., 'income', 'balance', 'cashflow', 'ratios', 'profile', 'yahoo_quarterly_financials'
            start_date: requested start
            end_date: requested end
            data: list/dict/DataFrame
            extra_meta: additional metadata

        Returns:
            version string if saved, else None
        """
        if data is None:
            return None

        # Normalize to DataFrame where possible; otherwise pickle raw
        obj_to_store: Any
        if isinstance(data, pd.DataFrame):
            obj_to_store = data
        elif isinstance(data, (list, dict)):
            try:
                obj_to_store = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
            except Exception:
                # Fallback to pickled raw object in DataFrame wrapper
                obj_to_store = pd.DataFrame({'raw': [json.dumps(data)]})
        else:
            # Wrap unknown types
            try:
                obj_to_store = pd.DataFrame({'raw_pickle': [pickle.dumps(data)]})
            except Exception:
                return None

        meta = {
            'source': source,
            'ticker': ticker,
            'payload': payload,
            'start_date': start_date,
            'end_date': end_date,
        }
        if extra_meta:
            try:
                meta.update(extra_meta)
            except Exception:
                pass

        data_type = f"raw_{source}_{payload}_{ticker or 'bulk'}_{start_date}_{end_date}"
        try:
            version = self.save_dataframe(obj_to_store, name=data_type, metadata=meta)
            return version
        except Exception as e:
            logger.warning(f"Failed to save raw payload {data_type}: {e}")
            return None

    def save_raw_yahoo_fundamentals(self, ticker: str, start_date: str, end_date: str,
                                     quarterly_financials: Optional[pd.DataFrame] = None,
                                     quarterly_balance_sheet: Optional[pd.DataFrame] = None,
                                     info: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
        """Convenience wrapper to save Yahoo raw fundamentals payloads."""
        results = {}
        results['quarterly_financials'] = self._save_raw_payload('Yahoo', ticker, 'quarterly_financials', start_date, end_date, quarterly_financials)
        results['quarterly_balance_sheet'] = self._save_raw_payload('Yahoo', ticker, 'quarterly_balance_sheet', start_date, end_date, quarterly_balance_sheet)
        results['info'] = self._save_raw_payload('Yahoo', ticker, 'info', start_date, end_date, info)
        return results

    def get_raw_fmp_payload(self, ticker: str, payload: str,
                             start_date: str, end_date: str,
                             search_limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Load previously saved FMP raw payload from data_objects, if it covers the requested range.

        Returns list[dict] or None when not found.
        """
        pattern = f"raw_FMP_{payload}_{ticker}_%"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT data_blob, metadata FROM data_objects
                   WHERE data_type LIKE ?
                   ORDER BY created_at DESC LIMIT ?''', (pattern, search_limit)
            )
            rows = cursor.fetchall()

        if not rows:
            return None

        req_start = pd.to_datetime(start_date)
        req_end = pd.to_datetime(end_date)

        for data_blob, meta_json in rows:
            try:
                meta = json.loads(meta_json) if meta_json else {}
                m_start = pd.to_datetime(meta.get('start_date')) if meta.get('start_date') else None
                m_end = pd.to_datetime(meta.get('end_date')) if meta.get('end_date') else None
                if m_start is not None and m_end is not None:
                    # check coverage (stored range should cover requested range)
                    if m_start <= req_start and m_end >= req_end:
                        df = pickle.loads(data_blob)
                        if isinstance(df, pd.DataFrame):
                            # convert to list of dicts compatible with endpoint returns
                            return df.to_dict(orient='records')
                        # fallback: if stored as raw json string in DataFrame
                        if isinstance(df, list):
                            return df
            except Exception:
                continue

        return None


# Global data store instance
_data_store = None
_data_store_config = {}

def get_data_store(base_dir: str = None) -> DataStore:
    """
    Get global data store instance.
    
    Args:
        base_dir: Base directory for data storage. If None, uses config.data.base_dir
        
    Returns:
        DataStore instance
        
    Note:
        The base_dir is read from config/settings.py (DATA_BASE_DIR environment variable).
        To change the database location, set the DATA_BASE_DIR environment variable
        in your .env file or system environment.
    """
    global _data_store

    from src.config.settings import get_config
    config = get_config()
    base_dir = config.data.base_dir
    
    if _data_store is None:
        _data_store = DataStore(base_dir)
        
    return _data_store


if __name__ == "__main__":
    # Intentionally left minimal: avoid fake I/O examples.
    logging.basicConfig(level=logging.INFO)
    store = get_data_store()
    stats = store.get_storage_stats()
    print(f"Database path: {stats['database_path']}")
    print(f"Records - price: {stats['price_records']}, fundamental: {stats['fundamental_records']}, data_objects: {stats.get('data_objects', 0)}, cache_entries: {stats.get('cache_entries', 0)}")
