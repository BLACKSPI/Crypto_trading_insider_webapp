import pandas as pd
import logging
from src.collectors.crypto_collector import CryptoDataCollector
import matplotlib.pyplot as plt
import seaborn as sns
import time
import ta
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe"""
    # Trend Indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['MACD'] = ta.trend.macd_diff(df['close'])
    
    # Momentum Indicators
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['Stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    
    # Volatility Indicators
    df['BB_high'] = ta.volatility.bollinger_hband(df['close'])
    df['BB_low'] = ta.volatility.bollinger_lband(df['close'])
    
    # Volume Indicators
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    return df

def plot_price_and_indicators(df: pd.DataFrame, pair: str):
    """Plot price and technical indicators"""
    # Set the style
    plt.style.use('default')
    sns.set_theme(style="darkgrid")
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Price and Bollinger Bands
    ax1.plot(df.index, df['close'], label='Price', color='blue', linewidth=2)
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', color='orange', linewidth=1.5)
    ax1.plot(df.index, df['BB_high'], label='BB Upper', color='gray', linestyle='--', alpha=0.7)
    ax1.plot(df.index, df['BB_low'], label='BB Lower', color='gray', linestyle='--', alpha=0.7)
    ax1.fill_between(df.index, df['BB_high'], df['BB_low'], color='gray', alpha=0.1)
    ax1.set_title(f'{pair} Price and Indicators', fontsize=12, pad=15)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.fill_between(df.index, 70, 100, color='r', alpha=0.1)
    ax2.fill_between(df.index, 0, 30, color='g', alpha=0.1)
    ax2.set_title('RSI', fontsize=12, pad=15)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Volume
    ax3.bar(df.index, df['volume'], label='Volume', color='green', alpha=0.5)
    ax3.set_title('Volume', fontsize=12, pad=15)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis dates
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{pair}_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_crypto_collector():
    """Test the cryptocurrency data collector"""
    collector = CryptoDataCollector()
    
    # Test each supported pair
    for pair in collector.supported_pairs:
        try:
            logger.info(f"\nTesting {pair}...")
            
            # Get current price
            current_price = collector.get_current_price(pair)
            logger.info(f"Current {pair} price: {current_price}")
            
            # Get historical data
            df = collector.get_historical_data(pair, period='30d', interval='1d')
            
            if not df.empty:
                # Add technical indicators
                df = add_technical_indicators(df)
                
                # Display basic statistics
                logger.info(f"\n{pair} Statistics:")
                logger.info(f"Data points: {len(df)}")
                logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
                logger.info(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
                logger.info(f"Average volume: {df['volume'].mean():.2f}")
                
                # Calculate additional metrics
                volatility = df['Returns'].std() * np.sqrt(252)  # Annualized volatility
                sharpe_ratio = (df['Returns'].mean() * 252) / volatility  # Annualized Sharpe ratio
                
                logger.info(f"Annualized Volatility: {volatility:.2%}")
                logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
                
                # Plot price and indicators
                plot_price_and_indicators(df, pair)
                logger.info(f"Plot saved as {pair}_analysis.png")
                
                # Display latest indicators
                latest = df.iloc[-1]
                logger.info(f"\nLatest indicators for {pair}:")
                logger.info(f"RSI: {latest['RSI']:.2f}")
                logger.info(f"MACD: {latest['MACD']:.2f}")
                logger.info(f"Bollinger Bands: {latest['BB_low']:.2f} - {latest['BB_high']:.2f}")
            
        except Exception as e:
            logger.error(f"Error testing {pair}: {str(e)}")
        
        # Wait between requests to avoid rate limits
        time.sleep(2)

if __name__ == "__main__":
    logger.info("Starting cryptocurrency data collector test...")
    test_crypto_collector()
    logger.info("Test completed.") 