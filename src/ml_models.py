import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
from typing import Dict, Tuple, List
import ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoMLModels:
    def __init__(self):
        """Initialize ML models and scalers"""
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.pattern_model = None
        self.anomaly_detector = None
        self.regime_classifier = None
        self.price_predictor = None
        self.min_data_points = 100  # Minimum data points required for training
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        if len(df) < self.min_data_points:
            logger.warning(f"Insufficient data points: {len(df)} < {self.min_data_points}")
            return pd.DataFrame()
            
        try:
            # Technical indicators
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            df['MACD'] = ta.trend.macd_diff(df['close'])
            df['BB_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['BB_lower'] = ta.volatility.bollinger_lband(df['close'])
            df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Price changes
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close']/df['close'].shift(1))
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_std'] = df['volume'].rolling(window=20).std()
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Market regime features
            df['trend'] = np.where(df['close'] > df['close'].rolling(window=20).mean(), 1, -1)
            df['volatility_regime'] = np.where(df['volatility'] > df['volatility'].rolling(window=20).mean(), 1, 0)
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def build_price_predictor(self, df: pd.DataFrame) -> None:
        """Build and train LSTM model for price prediction"""
        try:
            if len(df) < self.min_data_points:
                logger.warning("Insufficient data for training price predictor")
                return
                
            # Prepare data
            features = ['close', 'volume', 'RSI', 'MACD', 'ATR', 'volatility']
            data = df[features].values
            
            if len(data) == 0:
                logger.warning("No valid data for training")
                return
                
            # Scale data
            scaled_data = self.feature_scaler.fit_transform(data)
            
            # Create sequences
            X, y = [], []
            sequence_length = min(60, len(scaled_data) - 1)  # Adjust sequence length if needed
            
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, 0])  # Predict next close price
                
            X, y = np.array(X), np.array(y)
            
            if len(X) == 0:
                logger.warning("No sequences created for training")
                return
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
            
            self.price_predictor = model
            logger.info("Price prediction model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training price prediction model: {e}")
            self.price_predictor = None
    
    def predict_price(self, df: pd.DataFrame) -> Dict[str, float]:
        """Predict next price and confidence"""
        try:
            if len(df) < self.min_data_points:
                return {
                    'predicted_price': df['close'].iloc[-1],
                    'confidence': 0.0,
                    'current_price': df['close'].iloc[-1],
                    'predicted_change': 0.0
                }
            
            if self.price_predictor is None:
                self.build_price_predictor(df)
                if self.price_predictor is None:
                    return None
            
            # Prepare latest data
            features = ['close', 'volume', 'RSI', 'MACD', 'ATR', 'volatility']
            latest_data = df[features].values[-60:]  # Last 60 time steps
            scaled_data = self.feature_scaler.transform(latest_data)
            
            # Make prediction
            prediction = self.price_predictor.predict(np.array([scaled_data]), verbose=0)
            predicted_price = self.feature_scaler.inverse_transform(
                np.concatenate([prediction, np.zeros((1, len(features)-1))], axis=1)
            )[0, 0]
            
            # Calculate confidence (based on model's loss)
            confidence = 1 - self.price_predictor.evaluate(
                np.array([scaled_data]),
                np.array([scaled_data[-1, 0]]),
                verbose=0
            )
            
            return {
                'predicted_price': predicted_price,
                'confidence': confidence,
                'current_price': df['close'].iloc[-1],
                'predicted_change': ((predicted_price - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100
            }
            
        except Exception as e:
            logger.error(f"Error making price prediction: {e}")
            return None
    
    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect common chart patterns"""
        try:
            if len(df) < 20:  # Minimum required for pattern detection
                return []
                
            patterns = []
            
            # Double Top/Bottom
            for i in range(2, len(df)-2):
                # Double Top
                if (df['high'].iloc[i] > df['high'].iloc[i-1] and
                    df['high'].iloc[i] > df['high'].iloc[i-2] and
                    abs(df['high'].iloc[i] - df['high'].iloc[i-2]) < 0.02 * df['high'].iloc[i]):
                    patterns.append({
                        'type': 'Double Top',
                        'date': df.index[i],
                        'price': df['high'].iloc[i],
                        'confidence': 0.8
                    })
                
                # Double Bottom
                if (df['low'].iloc[i] < df['low'].iloc[i-1] and
                    df['low'].iloc[i] < df['low'].iloc[i-2] and
                    abs(df['low'].iloc[i] - df['low'].iloc[i-2]) < 0.02 * df['low'].iloc[i]):
                    patterns.append({
                        'type': 'Double Bottom',
                        'date': df.index[i],
                        'price': df['low'].iloc[i],
                        'confidence': 0.8
                    })
            
            # Head and Shoulders
            for i in range(3, len(df)-3):
                if (df['high'].iloc[i] > df['high'].iloc[i-1] and
                    df['high'].iloc[i] > df['high'].iloc[i-2] and
                    df['high'].iloc[i] > df['high'].iloc[i-3] and
                    df['high'].iloc[i-1] > df['high'].iloc[i-2] and
                    df['high'].iloc[i-3] > df['high'].iloc[i-2]):
                    patterns.append({
                        'type': 'Head and Shoulders',
                        'date': df.index[i],
                        'price': df['high'].iloc[i],
                        'confidence': 0.85
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect price and volume anomalies"""
        try:
            if len(df) < 20:  # Minimum required for anomaly detection
                return pd.DataFrame()
                
            # Prepare features for anomaly detection
            features = ['returns', 'volume', 'volatility']
            data = df[features].values
            
            if len(data) == 0:
                return pd.DataFrame()
                
            # Scale data
            scaled_data = self.feature_scaler.fit_transform(data)
            
            # Train isolation forest
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.anomaly_detector.fit(scaled_data)
            
            # Detect anomalies
            anomalies = self.anomaly_detector.predict(scaled_data)
            df['is_anomaly'] = anomalies
            
            return df[df['is_anomaly'] == -1]  # Return only anomalous points
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return pd.DataFrame()
    
    def classify_market_regime(self, df: pd.DataFrame) -> Dict:
        """Classify current market regime"""
        try:
            if len(df) < 20:  # Minimum required for regime classification
                return {
                    'regime': 'Unknown',
                    'confidence': 0.0,
                    'trend': 'Unknown',
                    'volatility': 'Unknown',
                    'momentum': 'Unknown'
                }
                
            # Calculate regime features
            df['trend'] = np.where(df['close'] > df['close'].rolling(window=20).mean(), 1, -1)
            df['volatility_regime'] = np.where(df['volatility'] > df['volatility'].rolling(window=20).mean(), 1, 0)
            df['momentum'] = np.where(df['RSI'] > 50, 1, -1)
            
            # Get latest regime indicators
            latest = df.iloc[-1]
            
            # Determine regime
            if latest['trend'] == 1 and latest['momentum'] == 1:
                regime = 'Bullish'
                confidence = 0.8
            elif latest['trend'] == -1 and latest['momentum'] == -1:
                regime = 'Bearish'
                confidence = 0.8
            elif latest['volatility_regime'] == 1:
                regime = 'High Volatility'
                confidence = 0.7
            else:
                regime = 'Sideways'
                confidence = 0.6
            
            return {
                'regime': regime,
                'confidence': confidence,
                'trend': 'Up' if latest['trend'] == 1 else 'Down',
                'volatility': 'High' if latest['volatility_regime'] == 1 else 'Low',
                'momentum': 'Positive' if latest['momentum'] == 1 else 'Negative'
            }
            
        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return {
                'regime': 'Unknown',
                'confidence': 0.0,
                'trend': 'Unknown',
                'volatility': 'Unknown',
                'momentum': 'Unknown'
            } 