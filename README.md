# Crypto Trading Dashboard

A real-time cryptocurrency trading dashboard with technical analysis, machine learning predictions, and risk management features.

## Features

- Real-time price charts with candlestick patterns
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Trading signals and entry/exit points
- Risk management metrics
- Machine learning predictions
- Market regime analysis
- Pattern recognition
- Anomaly detection

## Local Development

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
python src/visualization/dashboard.py
```

4. Open your browser and navigate to: http://localhost:8050

## Deployment

This project can be deployed for free on Render.com:

1. Create a Render account at https://render.com
2. Create a new Web Service
3. Connect your GitHub repository
4. Use the following settings:
   - Name: crypto-trading-dashboard
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn src.visualization.dashboard:server`
   - Plan: Free

## Project Structure

```
src/
├── visualization/
│   └── dashboard.py      # Main dashboard application
├── collectors/
│   └── crypto_collector.py  # Data collection module
├── ml_models.py          # Machine learning models
└── risk_management.py    # Risk management module
```

## Dependencies

- Dash: Web application framework
- Plotly: Interactive charts
- Pandas: Data manipulation
- TA: Technical analysis indicators
- YFinance: Market data
- Scikit-learn: Machine learning
- Gunicorn: Production server

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This project is for educational purposes only. Cryptocurrency trading involves significant risk. Always do your own research and never invest more than you can afford to lose. 