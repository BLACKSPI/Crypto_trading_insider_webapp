import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Optional
import numpy as np
import time
import ta  # Technical Analysis library

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataCollector:
    """
    A data collector for cryptocurrency price data using the Binance API.
    Supports multiple cryptocurrencies and timeframes with technical analysis capabilities.
    """
    def __init__(self):
        # Supported cryptocurrency pairs
        self.supported_pairs = [
            'BTCUSDT',   # Bitcoin
            'ETHUSDT',   # Ethereum
            'BNBUSDT',   # Binance Coin
            'ADAUSDT',   # Cardano
            'SOLUSDT',   # Solana
            'DOTUSDT',   # Polkadot
            'DOGEUSDT',  # Dogecoin
            'XRPUSDT',   # Ripple
            'MATICUSDT'  # Polygon
        ]
        # Supported timeframes with their Binance equivalents
        self.supported_timeframes = {
            '1d': '1d',    # Daily
            '4h': '4h',    # 4 hours
            '1h': '1h',    # 1 hour
            '15m': '15m',  # 15 minutes
            '5m': '5m'     # 5 minutes
        }
        self.base_url = 'https://api.binance.com/api/v3'
        self.retry_count = 3
        self.retry_delay = 10  # seconds
        logger.info("Initialized CryptoDataCollector with Binance API")

    def _make_api_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Make API request to Binance with retry logic.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters for the request
            
        Returns:
            dict: JSON response from the API
        """
        for attempt in range(self.retry_count):
            try:
                url = f"{self.base_url}/{endpoint}"
                logger.info(f"Requesting {url}")
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    logger.warning("Rate limit hit, waiting before retry...")
                    time.sleep(self.retry_delay * 2)
                else:
                    logger.error(f"HTTP error: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Exception during API request: {str(e)}")
            
            if attempt < self.retry_count - 1:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        return {}

    def get_historical_data(
        self,
        pair: str = 'BTCUSDT',
        period: str = '5d',
        interval: str = '1d',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical cryptocurrency price data using Binance API.
        
        Args:
            pair: Cryptocurrency pair (e.g., 'BTCUSDT', 'ETHUSDT')
            period: Time period to fetch (e.g., '1d', '5d', '1mo')
            interval: Data interval ('1d', '4h', '1h', '15m', '5m')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Historical price data with OHLCV and technical indicators
        """
        if pair not in self.supported_pairs:
            raise ValueError(f"Unsupported pair: {pair}. Supported pairs are: {', '.join(self.supported_pairs)}")
            
        if interval not in self.supported_timeframes:
            raise ValueError(f"Unsupported interval: {interval}. Supported intervals are: {', '.join(self.supported_timeframes.keys())}")
        
        # Convert interval to Binance format
        binance_interval = self.supported_timeframes[interval]
        
        # Calculate timestamps
        end_time = int(datetime.now().timestamp() * 1000)
        if end_date:
            end_time = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        if start_date:
            start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            # Default to 'period' days ago
            try:
                days = int(period.replace('d', ''))
            except Exception:
                days = 5
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        params = {
            'symbol': pair,
            'interval': binance_interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        
        data = self._make_api_request('klines', params)
        if data:
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Add additional useful columns
            df['Returns'] = df['close'].pct_change()
            df['Log_Returns'] = np.log(df['close']/df['close'].shift(1))
            
            logger.info(f"Retrieved {len(df)} data points for {pair}")
            return df
        
        logger.warning(f"No historical data available for {pair}")
        return pd.DataFrame()

    def get_current_price(self, pair: str = 'BTCUSDT') -> float:
        """
        Get the current cryptocurrency price using Binance API.
        
        Args:
            pair: Cryptocurrency pair (e.g., 'BTCUSDT', 'ETHUSDT')
            
        Returns:
            float: Current price of the cryptocurrency
        """
        if pair not in self.supported_pairs:
            raise ValueError(f"Unsupported pair: {pair}. Supported pairs are: {', '.join(self.supported_pairs)}")
        
        params = {'symbol': pair}
        data = self._make_api_request('ticker/price', params)
        
        if data and 'price' in data:
            price = float(data['price'])
            logger.info(f"Current {pair} price: {price}")
            return price
        
        logger.warning(f"No current price data available for {pair}")
        return 0.0

    def get_supported_pairs(self) -> List[str]:
        """Get list of supported cryptocurrency pairs."""
        return self.supported_pairs

    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes."""
        return list(self.supported_timeframes.keys()) 