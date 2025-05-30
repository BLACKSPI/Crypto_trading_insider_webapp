import logging
import schedule
import time
from datetime import datetime, timedelta
import pandas as pd
import os
import sys

from collectors.forex_collector import ForexDataCollector
from indicators.technical_indicators import TechnicalIndicators
from signals.signal_generator import SignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForexPipeline:
    def __init__(self):
        self.collector = ForexDataCollector()
        self.indicators = TechnicalIndicators()
        self.signal_generator = SignalGenerator()
        self.data_dir = 'data'
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def process_pair(self, pair: str):
        """
        Process a single currency pair.
        
        Args:
            pair: Currency pair symbol
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Processing {pair} (attempt {attempt + 1}/{self.max_retries})")
                
                # Get historical data with a longer period to ensure we have enough data
                df = self.collector.get_historical_data(
                    pair,
                    period='5d',  # Increased from 1d to 5d
                    interval='1d'  # Changed from 1h to 1d
                )
                
                if df.empty:
                    logger.warning(f"No data available for {pair}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return
                
                # Add technical indicators
                df = self.indicators.add_all_indicators(df)
                
                # Generate signals
                df = self.signal_generator.generate_signals(df)
                
                # Save processed data
                filename = f"{self.data_dir}/{pair.replace('=', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename)
                logger.info(f"Saved data to {filename}")
                
                # Get and log signal summary
                signal_summary = self.signal_generator.get_signal_summary(df)
                logger.info(f"Signal summary for {pair}: {signal_summary}")
                
                # If we get here, processing was successful
                return
                
            except Exception as e:
                logger.error(f"Error processing {pair} (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
    
    def run_pipeline(self):
        """Run the pipeline for all supported pairs"""
        logger.info("Starting forex pipeline")
        
        successful_pairs = []
        failed_pairs = []
        
        for pair in self.collector.supported_pairs:
            try:
                self.process_pair(pair)
                successful_pairs.append(pair)
            except Exception as e:
                logger.error(f"Failed to process {pair}: {str(e)}")
                failed_pairs.append(pair)
        
        logger.info(f"Pipeline completed. Successful: {len(successful_pairs)}, Failed: {len(failed_pairs)}")
        if failed_pairs:
            logger.warning(f"Failed pairs: {', '.join(failed_pairs)}")

def main():
    try:
        pipeline = ForexPipeline()
        
        # Run immediately on startup
        pipeline.run_pipeline()
        
        # Schedule regular runs (every 15 minutes)
        schedule.every(15).minutes.do(pipeline.run_pipeline)
        
        logger.info("Pipeline started successfully. Press Ctrl+C to stop.")
        logger.info("Data will be updated every 15 minutes.")
        
        # Keep the script running with more frequent checks
        while True:
            schedule.run_pending()
            time.sleep(1)  # Check every second for pending tasks
            
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 