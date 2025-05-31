import yfinance as yf
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gold_data():
    try:
        # Test different gold tickers
        tickers = ['GC=F', 'XAUUSD=X', 'XAU=X']
        
        for ticker in tickers:
            logger.info(f"\nTesting ticker: {ticker}")
            
            # Get the ticker object
            ticker_obj = yf.Ticker(ticker)
            
            # Try to get info
            try:
                info = ticker_obj.info
                logger.info(f"Info available: {bool(info)}")
            except Exception as e:
                logger.error(f"Error getting info: {str(e)}")
            
            # Try to get historical data
            try:
                hist = ticker_obj.history(period='1d')
                logger.info(f"Historical data shape: {hist.shape}")
                logger.info(f"Columns: {hist.columns.tolist()}")
                logger.info(f"Latest price: {hist['Close'].iloc[-1] if not hist.empty else 'No data'}")
            except Exception as e:
                logger.error(f"Error getting historical data: {str(e)}")
            
            logger.info("-" * 50)
    
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting yfinance test...")
    test_gold_data()
    logger.info("Test completed.") 