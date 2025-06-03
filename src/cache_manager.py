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
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is too old
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(minutes=max_age_minutes):
                return None
                
            # Convert data back to DataFrame
            df = pd.DataFrame(cache_data['data'])
            df.index = pd.to_datetime(df.index)
            return df
            
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")
            return None
            
    def cache_data(self, key, df):
        """Cache DataFrame with timestamp"""
        cache_path = self._get_cache_path(key)
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': df.to_dict(orient='dict')
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
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