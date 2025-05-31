import pandas as pd
import numpy as np
import ta

class TechnicalAnalyzer:
    def __init__(self):
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Calculate RSI
        df['RSI'] = ta.momentum.RSIIndicator(
            close=df['Close'],
            window=self.rsi_period
        ).rsi()
        
        # Calculate MACD
        macd = ta.trend.MACD(
            close=df['Close'],
            window_slow=self.macd_slow,
            window_fast=self.macd_fast,
            window_sign=self.macd_signal
        )
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=df['Close'],
            window=self.bb_period,
            window_dev=self.bb_std
        )
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        
        return df
    
    def generate_trading_signals(self, df: pd.DataFrame) -> dict:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary containing trading signals
        """
        if df.empty:
            return {
                'RSI_Signal': 'Neutral',
                'MACD_Signal': 'Neutral',
                'BB_Signal': 'Neutral'
            }
        
        # Get the latest values
        latest = df.iloc[-1]
        
        # RSI Signal
        rsi_signal = 'Neutral'
        if latest['RSI'] > 70:
            rsi_signal = 'Sell'
        elif latest['RSI'] < 30:
            rsi_signal = 'Buy'
        
        # MACD Signal
        macd_signal = 'Neutral'
        if latest['MACD'] > latest['MACD_Signal']:
            macd_signal = 'Buy'
        elif latest['MACD'] < latest['MACD_Signal']:
            macd_signal = 'Sell'
        
        # Bollinger Bands Signal
        bb_signal = 'Neutral'
        if latest['Close'] > latest['BB_Upper']:
            bb_signal = 'Sell'
        elif latest['Close'] < latest['BB_Lower']:
            bb_signal = 'Buy'
        
        return {
            'RSI_Signal': rsi_signal,
            'MACD_Signal': macd_signal,
            'BB_Signal': bb_signal
        } 