import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class SignalGenerator:
    def __init__(self):
        self.signal_weights = {
            'rsi': 0.2,
            'macd': 0.3,
            'bollinger': 0.2,
            'stochastic': 0.2,
            'moving_averages': 0.1
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on multiple technical indicators.
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with added signal columns
        """
        # Generate individual signals
        df['RSI_Signal'] = self._generate_rsi_signal(df)
        df['MACD_Signal'] = self._generate_macd_signal(df)
        df['BB_Signal'] = self._generate_bollinger_signal(df)
        df['Stoch_Signal'] = self._generate_stochastic_signal(df)
        df['MA_Signal'] = self._generate_ma_signal(df)
        
        # Calculate weighted signal
        df['Weighted_Signal'] = (
            df['RSI_Signal'] * self.signal_weights['rsi'] +
            df['MACD_Signal'] * self.signal_weights['macd'] +
            df['BB_Signal'] * self.signal_weights['bollinger'] +
            df['Stoch_Signal'] * self.signal_weights['stochastic'] +
            df['MA_Signal'] * self.signal_weights['moving_averages']
        )
        
        # Generate final signal
        df['Final_Signal'] = df['Weighted_Signal'].apply(self._convert_to_signal)
        
        return df
    
    def _generate_rsi_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate RSI-based signals"""
        signal = pd.Series(0, index=df.index)
        signal[df['RSI'] < 30] = 1  # Oversold - Buy
        signal[df['RSI'] > 70] = -1  # Overbought - Sell
        return signal
    
    def _generate_macd_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate MACD-based signals"""
        signal = pd.Series(0, index=df.index)
        signal[df['MACD'] > df['MACD_Signal']] = 1  # Bullish crossover
        signal[df['MACD'] < df['MACD_Signal']] = -1  # Bearish crossover
        return signal
    
    def _generate_bollinger_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate Bollinger Bands-based signals"""
        signal = pd.Series(0, index=df.index)
        signal[df['Close'] < df['BB_Lower']] = 1  # Price below lower band - Buy
        signal[df['Close'] > df['BB_Upper']] = -1  # Price above upper band - Sell
        return signal
    
    def _generate_stochastic_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate Stochastic Oscillator-based signals"""
        signal = pd.Series(0, index=df.index)
        signal[(df['Stoch_K'] < 20) & (df['Stoch_D'] < 20)] = 1  # Oversold
        signal[(df['Stoch_K'] > 80) & (df['Stoch_D'] > 80)] = -1  # Overbought
        return signal
    
    def _generate_ma_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate Moving Average-based signals"""
        signal = pd.Series(0, index=df.index)
        # Golden Cross (50-day crosses above 200-day)
        signal[(df['SMA_50'] > df['SMA_200']) & 
               (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))] = 1
        # Death Cross (50-day crosses below 200-day)
        signal[(df['SMA_50'] < df['SMA_200']) & 
               (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))] = -1
        return signal
    
    def _convert_to_signal(self, value: float) -> int:
        """Convert weighted signal to final signal"""
        if value > 0.2:
            return SignalType.BUY.value
        elif value < -0.2:
            return SignalType.SELL.value
        return SignalType.HOLD.value
    
    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate the strength of the current signal.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            Signal strength as float (-1 to 1)
        """
        if df.empty:
            return 0.0
        
        latest_signal = df['Weighted_Signal'].iloc[-1]
        return latest_signal
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get a summary of the current trading signals.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            Dictionary with signal summary
        """
        if df.empty:
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'indicators': {}
            }
        
        latest = df.iloc[-1]
        
        summary = {
            'signal': SignalType(latest['Final_Signal']).name,
            'strength': abs(latest['Weighted_Signal']),
            'indicators': {
                'RSI': {
                    'value': latest['RSI'],
                    'signal': SignalType(latest['RSI_Signal']).name
                },
                'MACD': {
                    'value': latest['MACD'],
                    'signal': SignalType(latest['MACD_Signal']).name
                },
                'Bollinger Bands': {
                    'signal': SignalType(latest['BB_Signal']).name
                },
                'Stochastic': {
                    'value': latest['Stoch_K'],
                    'signal': SignalType(latest['Stoch_Signal']).name
                },
                'Moving Averages': {
                    'signal': SignalType(latest['MA_Signal']).name
                }
            }
        }
        
        return summary 