import pandas as pd
import json
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def _get_cache_path(self, key):
        """Get the cache file path for a given key"""
        return os.path.join(self.cache_dir, f"{key}.json")
        
    def get_cached_data(self, key, max_age_minutes=5):
        """Get cached data if it exists and is not too old"""
        cache_path = self._get_cache_path(key)
        pickle_path = cache_path + '.pkl'
        
        # Try pickle file first
        if os.path.exists(pickle_path):
            try:
                df = pd.read_pickle(pickle_path)
                file_time = datetime.fromtimestamp(os.path.getmtime(pickle_path))
                if datetime.now() - file_time > timedelta(minutes=max_age_minutes):
                    return None
                return df
            except Exception as e:
                logger.error(f"Error reading pickle cache: {str(e)}")
        
        # Try JSON file
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                    
                # Check if cache is too old
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cache_time > timedelta(minutes=max_age_minutes):
                    return None
                    
                # Convert data back to DataFrame
                data_dict = cache_data['data']
                df = pd.DataFrame.from_dict(data_dict, orient='index')
                df.index = pd.to_datetime(df.index)
                return df
                
            except Exception as e:
                logger.error(f"Error reading JSON cache: {str(e)}")
                return None
        
        return None
            
    def cache_data(self, key, df):
        """Cache DataFrame with timestamp"""
        cache_path = self._get_cache_path(key)
        
        try:
            # Save as pickle first (more reliable)
            df.to_pickle(cache_path + '.pkl')
            logger.info(f"Saved cache as pickle file: {cache_path}.pkl")
            
            # Also try to save as JSON
            try:
                data_dict = df.to_dict(orient='index')
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'data': data_dict
                }
                
                with open(cache_path, 'w') as f:
                    json.dump(cache_data, f, default=str)
                logger.info(f"Saved cache as JSON file: {cache_path}")
                    
            except Exception as json_error:
                logger.warning(f"Could not save JSON cache: {str(json_error)}")
                
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")
            
    def clear_old_cache(self, max_age_hours=24):
        """Clear cache files older than max_age_hours"""
        try:
            current_time = datetime.now()
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if current_time - file_time > timedelta(hours=max_age_hours):
                        os.remove(file_path)
                        
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}") 