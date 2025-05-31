import pandas as pd
import numpy as np
from typing import Union, Optional
import ta

class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        df = TechnicalIndicators.add_moving_averages(df)
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_stochastic(df)
        return df
    
    @staticmethod
    def add_moving_averages(
        df: pd.DataFrame,
        windows: list = [20, 50, 200]
    ) -> pd.DataFrame:
        """
        Add Simple Moving Averages (SMA) to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            windows: List of window sizes for SMAs
            
        Returns:
            DataFrame with added SMAs
        """
        for window in windows:
            df[f'SMA_{window}'] = ta.trend.sma_indicator(df['Close'], window=window)
        return df
    
    @staticmethod
    def add_rsi(
        df: pd.DataFrame,
        window: int = 14
    ) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            window: RSI window size
            
        Returns:
            DataFrame with added RSI
        """
        df['RSI'] = ta.momentum.rsi(df['Close'], window=window)
        return df
    
    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9
    ) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            window_slow: Slow EMA window
            window_fast: Fast EMA window
            window_sign: Signal line window
            
        Returns:
            DataFrame with added MACD
        """
        macd = ta.trend.MACD(
            df['Close'],
            window_slow=window_slow,
            window_fast=window_fast,
            window_sign=window_sign
        )
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        return df
    
    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        window: int = 20,
        window_dev: int = 2
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            window: Moving average window
            window_dev: Number of standard deviations
            
        Returns:
            DataFrame with added Bollinger Bands
        """
        bollinger = ta.volatility.BollingerBands(
            df['Close'],
            window=window,
            window_dev=window_dev
        )
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['BB_Lower'] = bollinger.bollinger_lband()
        return df
    
    @staticmethod
    def add_stochastic(
        df: pd.DataFrame,
        window: int = 14,
        smooth_window: int = 3
    ) -> pd.DataFrame:
        """
        Add Stochastic Oscillator to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            window: Stochastic window
            smooth_window: Smoothing window
            
        Returns:
            DataFrame with added Stochastic Oscillator
        """
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=window,
            smooth_window=smooth_window
        )
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        return df 