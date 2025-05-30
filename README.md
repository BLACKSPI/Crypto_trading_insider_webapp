# Cryptocurrency Trading Dashboard with ML Analysis

A comprehensive cryptocurrency trading dashboard that combines technical analysis, machine learning predictions, and risk management tools for informed trading decisions.

## Features

### Technical Analysis
- Real-time price charts with multiple timeframes (1h, 15m, 5m)
- Multiple technical indicators:
  - Moving Averages (SMA, EMA)
  - MACD with signal line and histogram
  - RSI with overbought/oversold levels
  - Bollinger Bands
  - Stochastic Oscillator
  - On-Balance Volume (OBV)
- Entry and exit signals based on indicator combinations
- Pattern recognition for common chart patterns

### Machine Learning Analysis
- Price prediction using LSTM neural networks
- Market regime classification
- Anomaly detection for unusual price/volume behavior
- Pattern recognition for technical patterns
- Confidence scoring for all predictions

### Risk Management
- Position size calculator
- Stop-loss and take-profit suggestions
- Risk-reward ratio calculator
- Portfolio allocation recommendations
- Risk metrics including:
  - Volatility analysis
  - Value at Risk (VaR)
  - Maximum loss/profit calculations

### Dashboard Features
- Interactive charts with zoom and pan capabilities
- Real-time data updates
- Multiple timeframe analysis
- Trading signals panel
- Recent trading points display
- Risk management panel
- ML analysis panel

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-trading-dashboard.git
cd crypto-trading-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
python src/visualization/dashboard.py
```

4. Access the dashboard at `http://127.0.0.1:8050`

## Dependencies

- pandas>=1.3.0
- numpy>=1.19.5
- requests>=2.26.0
- python-binance>=1.0.15
- dash>=2.0.0
- plotly>=5.3.1
- ta>=0.7.0
- scikit-learn>=0.24.2
- tensorflow>=2.8.0
- seaborn>=0.11.2
- matplotlib>=3.4.3

## Project Structure

```
crypto-trading-dashboard/
├── src/
│   ├── collectors/
│   │   └── crypto_collector.py
│   ├── visualization/
│   │   └── dashboard.py
│   ├── ml_models.py
│   └── risk_management.py
├── requirements.txt
└── README.md
```

## Usage

1. Select a cryptocurrency pair from the dropdown menu
2. Choose your preferred timeframe
3. Use the risk management panel to:
   - Set your account balance
   - Define risk per trade
   - Set stop-loss and take-profit levels
4. Monitor the ML analysis panel for:
   - Price predictions
   - Market regime classification
   - Pattern recognition
   - Anomaly detection
5. Use the trading signals panel to identify potential entry and exit points

## Features in Detail

### Technical Analysis
The dashboard provides comprehensive technical analysis tools:
- Multiple timeframe analysis (1h, 15m, 5m)
- Real-time candlestick charts
- Technical indicators with customizable parameters
- Entry/exit signals based on multiple indicators
- Volume analysis

### Machine Learning Features
Advanced ML capabilities for enhanced trading decisions:
- LSTM-based price prediction
- Market regime classification (Bullish, Bearish, High Volatility, Sideways)
- Pattern recognition for common chart patterns
- Anomaly detection for unusual market behavior
- Confidence scoring for all predictions

### Risk Management
Comprehensive risk management tools:
- Position size calculation based on account balance and risk tolerance
- Stop-loss and take-profit suggestions
- Risk-reward ratio calculation
- Portfolio allocation recommendations
- Risk metrics including volatility and VaR

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This project is for educational purposes only. Cryptocurrency trading involves significant risk. Always do your own research and never invest more than you can afford to lose. 