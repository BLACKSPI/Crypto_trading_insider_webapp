import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, account_balance: float = 10000.0, risk_per_trade: float = 0.02):
        """
        Initialize the RiskManager with account settings.
        
        Args:
            account_balance: Total account balance in USDT
            risk_per_trade: Maximum risk per trade as a decimal (e.g., 0.02 for 2%)
        """
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.positions = {}  # Track open positions
        
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_amount: float = None
    ) -> Dict[str, float]:
        """
        Calculate the position size based on risk parameters.
        
        Args:
            entry_price: Current price of the asset
            stop_loss: Stop loss price
            risk_amount: Optional specific risk amount (if None, uses account_balance * risk_per_trade)
            
        Returns:
            Dict containing position size and risk metrics
        """
        if risk_amount is None:
            risk_amount = self.account_balance * self.risk_per_trade
            
        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            raise ValueError("Entry price and stop loss cannot be the same")
            
        position_size = risk_amount / price_risk
        position_value = position_size * entry_price
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_percentage': (risk_amount / self.account_balance) * 100
        }
    
    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        atr: float,
        risk_reward_ratio: float = 2.0,
        trend: str = 'up'
    ) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels based on ATR and risk-reward ratio.
        
        Args:
            entry_price: Current price of the asset
            atr: Average True Range value
            risk_reward_ratio: Desired risk-reward ratio
            trend: 'up' for long positions, 'down' for short positions
            
        Returns:
            Dict containing stop loss and take profit levels
        """
        if trend == 'up':
            stop_loss = entry_price - (atr * 1.5)
            take_profit = entry_price + (atr * 1.5 * risk_reward_ratio)
        else:
            stop_loss = entry_price + (atr * 1.5)
            take_profit = entry_price - (atr * 1.5 * risk_reward_ratio)
            
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': abs(entry_price - stop_loss),
            'reward_amount': abs(take_profit - entry_price),
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def calculate_portfolio_allocation(
        self,
        assets: List[str],
        prices: Dict[str, float],
        risk_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate optimal portfolio allocation based on risk scores.
        
        Args:
            assets: List of asset symbols
            prices: Dictionary of current prices
            risk_scores: Dictionary of risk scores (0-1, where 1 is highest risk)
            
        Returns:
            Dict containing allocation percentages for each asset
        """
        # Normalize risk scores
        total_risk = sum(risk_scores.values())
        if total_risk == 0:
            raise ValueError("Risk scores cannot all be zero")
            
        # Calculate inverse risk weights (lower risk = higher allocation)
        weights = {asset: (1 - score) / len(assets) for asset, score in risk_scores.items()}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        allocations = {asset: (weight / total_weight) * 100 for asset, weight in weights.items()}
        
        return allocations
    
    def calculate_risk_metrics(
        self,
        historical_data: pd.DataFrame,
        position_size: float,
        stop_loss: float,
        take_profit: float
    ) -> Dict[str, float]:
        """
        Calculate various risk metrics for a trade.
        
        Args:
            historical_data: DataFrame with OHLCV data
            position_size: Size of the position
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Dict containing risk metrics
        """
        # Calculate volatility
        returns = historical_data['close'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(returns, 5) * position_size
        
        # Calculate potential profit and loss
        max_loss = (stop_loss - historical_data['close'].iloc[-1]) * position_size
        max_profit = (take_profit - historical_data['close'].iloc[-1]) * position_size
        
        return {
            'volatility': volatility * 100,  # as percentage
            'var_95': abs(var_95),
            'max_loss': abs(max_loss),
            'max_profit': max_profit,
            'risk_reward_ratio': abs(max_profit / max_loss) if max_loss != 0 else 0
        }
    
    def update_account_balance(self, new_balance: float):
        """Update the account balance."""
        self.account_balance = new_balance
        
    def add_position(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        stop_loss: float,
        take_profit: float
    ):
        """Add a new position to track."""
        self.positions[symbol] = {
            'entry_price': entry_price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'unrealized_pnl': 0.0
        }
        
    def update_position(self, symbol: str, current_price: float):
        """Update position with current price."""
        if symbol in self.positions:
            position = self.positions[symbol]
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['position_size']
            
    def get_position_summary(self) -> Dict[str, Dict]:
        """Get summary of all positions."""
        return self.positions 