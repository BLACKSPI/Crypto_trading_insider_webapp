import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import sys
import logging
import ta

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.crypto_collector import CryptoDataCollector
from risk_management import RiskManager
from ml_models import CryptoMLModels
from cache_manager import CacheManager

# Set up logging with more detailed configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
logger.info("Initializing dashboard components...")
collector = CryptoDataCollector()
risk_manager = RiskManager()
ml_models = CryptoMLModels()
cache_manager = CacheManager()
logger.info("Dashboard components initialized successfully")

def identify_trading_points(df: pd.DataFrame) -> pd.DataFrame:
    """Identify entry and exit points based on technical indicators"""
    # Initialize columns for trading points
    df['Entry'] = False
    df['Exit'] = False
    
    # RSI-based signals
    df.loc[(df['RSI'] < 30) & (df['RSI'].shift(1) >= 30), 'Entry'] = True  # RSI crosses below 30
    df.loc[(df['RSI'] > 70) & (df['RSI'].shift(1) <= 70), 'Exit'] = True   # RSI crosses above 70
    
    # MACD-based signals
    df.loc[(df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)), 'Entry'] = True  # MACD crosses above signal
    df.loc[(df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)), 'Exit'] = True   # MACD crosses below signal
    
    # Bollinger Bands signals
    df.loc[(df['close'] < df['BB_Lower']) & (df['close'].shift(1) >= df['BB_Lower'].shift(1)), 'Entry'] = True  # Price crosses below lower band
    df.loc[(df['close'] > df['BB_Upper']) & (df['close'].shift(1) <= df['BB_Upper'].shift(1)), 'Exit'] = True   # Price crosses above upper band
    
    # Stochastic signals
    df.loc[(df['Stoch_K'] < 20) & (df['Stoch_K'].shift(1) >= 20), 'Entry'] = True  # Stochastic crosses below 20
    df.loc[(df['Stoch_K'] > 80) & (df['Stoch_K'].shift(1) <= 80), 'Exit'] = True   # Stochastic crosses above 80
    
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators and trading signals"""
    # Trend Indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Lower'] = bollinger.bollinger_lband()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Volume Indicators
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # Identify trading points
    df = identify_trading_points(df)
    
    return df

def generate_trading_signals(df: pd.DataFrame) -> dict:
    """Generate trading signals summary"""
    latest = df.iloc[-1]
    signals = {
        'RSI': 'Buy' if latest['RSI'] < 30 else 'Sell' if latest['RSI'] > 70 else 'Neutral',
        'MACD': 'Buy' if latest['MACD'] > latest['MACD_Signal'] else 'Sell',
        'BB': 'Buy' if latest['close'] < latest['BB_Lower'] else 'Sell' if latest['close'] > latest['BB_Upper'] else 'Neutral',
        'Stoch': 'Buy' if latest['Stoch_K'] < 20 else 'Sell' if latest['Stoch_K'] > 80 else 'Neutral'
    }
    
    # Overall signal
    buy_signals = sum(1 for signal in signals.values() if signal == 'Buy')
    sell_signals = sum(1 for signal in signals.values() if signal == 'Sell')
    
    if buy_signals > sell_signals:
        signals['Overall'] = 'Buy'
    elif sell_signals > buy_signals:
        signals['Overall'] = 'Sell'
    else:
        signals['Overall'] = 'Neutral'
    
    return signals

def get_recent_trading_points(df: pd.DataFrame, n_points: int = 5) -> list:
    """Get the most recent trading points"""
    recent_points = []
    
    # Get the last n_points entries and exits
    entries = df[df['Entry']].tail(n_points)
    exits = df[df['Exit']].tail(n_points)
    
    for idx, row in entries.iterrows():
        recent_points.append({
            'date': idx,
            'type': 'Entry',
            'action': 'BUY',  # Entry points are always BUY actions
            'price': row['close'],
            'reason': 'Multiple indicators suggest buying opportunity'
        })
    
    for idx, row in exits.iterrows():
        recent_points.append({
            'date': idx,
            'type': 'Exit',
            'action': 'SELL',  # Exit points are always SELL actions
            'price': row['close'],
            'reason': 'Multiple indicators suggest selling opportunity'
        })
    
    # Sort by date
    recent_points.sort(key=lambda x: x['date'], reverse=True)
    return recent_points[:n_points]

# Create the Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1('Crypto Trading Dashboard', style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    # Controls
    html.Div([
        html.Div([
            html.Label('Select Crypto Pair:', style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(
                id='pair-selector',
                options=[{'label': pair, 'value': pair} for pair in collector.get_supported_pairs()],
                value=collector.get_supported_pairs()[0],
                style={'width': '100%', 'marginBottom': '20px'}
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label('Select Timeframe:', style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(
                id='timeframe-selector',
                options=[{'label': tf, 'value': tf} for tf in collector.get_supported_timeframes()],
                value='1h',
                style={'width': '100%', 'marginBottom': '20px'}
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
    ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    # ML Analysis Panel
    html.Div([
        html.H3('Machine Learning Analysis', style={'color': '#2c3e50', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H4('Price Prediction', style={'color': '#2c3e50'}),
                html.Div(id='price-prediction', style={'padding': '10px'})
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
            
            html.Div([
                html.H4('Market Regime', style={'color': '#2c3e50'}),
                html.Div(id='market-regime', style={'padding': '10px'})
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
            
            html.Div([
                html.H4('Pattern Recognition', style={'color': '#2c3e50'}),
                html.Div(id='pattern-recognition', style={'padding': '10px'})
            ], style={'width': '30%', 'display': 'inline-block'})
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.H4('Anomaly Detection', style={'color': '#2c3e50'}),
            html.Div(id='anomaly-detection', style={'padding': '10px'})
        ])
    ], style={
        'padding': '20px',
        'backgroundColor': '#ffffff',
        'borderRadius': '10px',
        'marginBottom': '30px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Risk Management Panel
    html.Div([
        html.H3('Risk Management', style={'color': '#2c3e50', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.Label('Account Balance (USDT):', style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='account-balance',
                    type='number',
                    value=10000,
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
                html.Label('Risk Per Trade (%):', style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='risk-per-trade',
                    type='number',
                    value=2,
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
                html.Button('Update Risk Settings', id='update-risk-settings', n_clicks=0,
                           style={'backgroundColor': '#2c3e50', 'color': 'white', 'border': 'none', 'padding': '10px'})
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
            
            html.Div([
                html.Label('Stop Loss (%):', style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='stop-loss',
                    type='number',
                    value=2,
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
                html.Label('Take Profit (%):', style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='take-profit',
                    type='number',
                    value=4,
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
                html.Button('Calculate Position Size', id='calculate-position', n_clicks=0,
                           style={'backgroundColor': '#2c3e50', 'color': 'white', 'border': 'none', 'padding': '10px'})
            ], style={'width': '30%', 'display': 'inline-block'})
        ], style={'marginBottom': '20px'}),
        
        # Risk metrics display
        html.Div(id='risk-metrics', style={
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '10px',
            'marginTop': '20px'
        })
    ], style={
        'padding': '20px',
        'backgroundColor': '#ffffff',
        'borderRadius': '10px',
        'marginBottom': '30px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Main chart
    dcc.Graph(id='price-chart', style={'height': '600px', 'marginBottom': '30px'}),
    
    # Indicators chart
    dcc.Graph(id='indicators-chart', style={'height': '400px', 'marginBottom': '30px'}),
    
    # Trading signals panel
    html.Div(id='signals-panel', style={
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '10px',
        'marginTop': '20px'
    }),
    
    # Recent trading points panel
    html.Div(id='trading-points-panel', style={
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '10px',
        'marginTop': '20px'
    }),
    
    # Auto-refresh interval
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
])

@app.callback(
    [Output('price-chart', 'figure'),
     Output('indicators-chart', 'figure'),
     Output('signals-panel', 'children'),
     Output('trading-points-panel', 'children'),
     Output('risk-metrics', 'children'),
     Output('price-prediction', 'children'),
     Output('market-regime', 'children'),
     Output('pattern-recognition', 'children'),
     Output('anomaly-detection', 'children')],
    [Input('pair-selector', 'value'),
     Input('timeframe-selector', 'value'),
     Input('interval-component', 'n_intervals'),
     Input('calculate-position', 'n_clicks'),
     Input('update-risk-settings', 'n_clicks')],
    [State('account-balance', 'value'),
     State('risk-per-trade', 'value'),
     State('stop-loss', 'value'),
     State('take-profit', 'value')]
)
def update_dashboard(pair, timeframe, n, n_clicks_calc, n_clicks_update, balance, risk_percent, sl_percent, tp_percent):
    try:
        logger.info(f"Updating dashboard for {pair} with {timeframe} timeframe")
        
        # Try to get cached data first
        cache_key = f"{pair}_{timeframe}"
        logger.debug(f"Attempting to get cached data for {cache_key}")
        df = cache_manager.get_cached_data(cache_key)
        
        # If no cached data or it's too old, fetch fresh data
        if df is None:
            logger.info(f"No valid cache found for {cache_key}, fetching fresh data")
            df = collector.get_historical_data(pair=pair, period='30d', interval=timeframe)
            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} data points for {pair}")
                # Cache the fresh data
                cache_manager.cache_data(cache_key, df)
                logger.debug(f"Cached fresh data for {cache_key}")
            else:
                logger.warning(f"No data received for {pair}")
        
        if df.empty:
            logger.error(f"Empty dataframe for {pair}")
            return {}, {}, 'No data available.', 'No data available.', html.Div("No data available."), 'No data available.', 'No data available.', 'No data available.', 'No data available.'
        
        # Prepare data for ML models
        logger.info("Preparing features for ML models")
        df = ml_models.prepare_features(df)
        if df.empty:
            logger.error("Feature preparation resulted in empty dataframe")
            return {}, {}, 'Error preparing features.', 'Error preparing features.', html.Div("Error preparing features."), 'Error', 'Error', 'Error', 'Error'
        
        # Calculate indicators and identify trading points
        logger.info("Calculating technical indicators")
        df = calculate_indicators(df)
        
        # Get ML predictions and analysis (with caching)
        ml_cache_key = f"{pair}_{timeframe}_ml"
        logger.debug(f"Attempting to get cached ML results for {ml_cache_key}")
        ml_data = cache_manager.get_cached_data(ml_cache_key, max_age_minutes=15)
        
        if ml_data is None:
            logger.info("No valid ML cache found, computing new predictions")
            price_prediction = ml_models.predict_price(df)
            market_regime = ml_models.classify_market_regime(df)
            patterns = ml_models.detect_patterns(df)
            anomalies = ml_models.detect_anomalies(df)
            
            # Cache ML results
            ml_results = {
                'price_prediction': price_prediction,
                'market_regime': market_regime,
                'patterns': patterns,
                'anomalies': anomalies
            }
            cache_manager.cache_data(ml_cache_key, pd.DataFrame([ml_results]))
            logger.debug(f"Cached ML results for {ml_cache_key}")
        else:
            logger.info("Using cached ML results")
            ml_results = ml_data.iloc[0].to_dict()
            price_prediction = ml_results['price_prediction']
            market_regime = ml_results['market_regime']
            patterns = ml_results['patterns']
            anomalies = ml_results['anomalies']
        
        # Create price prediction display
        if price_prediction:
            price_prediction_display = html.Div([
                html.P(f"Predicted Price: ${price_prediction['predicted_price']:.2f}"),
                html.P(f"Predicted Change: {price_prediction['predicted_change']:.2f}%"),
                html.P(f"Confidence: {price_prediction['confidence']*100:.1f}%")
            ])
        else:
            price_prediction_display = html.Div([
                html.P("Insufficient data for price prediction"),
                html.P("Collecting more data...")
            ])
        
        # Create market regime display
        if market_regime and market_regime['regime'] != 'Unknown':
            market_regime_display = html.Div([
                html.P(f"Current Regime: {market_regime['regime']}"),
                html.P(f"Confidence: {market_regime['confidence']*100:.1f}%"),
                html.P(f"Trend: {market_regime['trend']}"),
                html.P(f"Volatility: {market_regime['volatility']}"),
                html.P(f"Momentum: {market_regime['momentum']}")
            ])
        else:
            market_regime_display = html.Div([
                html.P("Insufficient data for market regime classification"),
                html.P("Collecting more data...")
            ])
        
        # Create pattern recognition display
        if patterns:
            pattern_display = html.Div([
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th('Pattern'),
                            html.Th('Date'),
                            html.Th('Price'),
                            html.Th('Confidence')
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(pattern['type']),
                            html.Td(pattern['date'].strftime('%Y-%m-%d %H:%M')),
                            html.Td(f"${pattern['price']:.2f}"),
                            html.Td(f"{pattern['confidence']*100:.1f}%")
                        ]) for pattern in patterns[-5:]  # Show last 5 patterns
                    ])
                ])
            ])
        else:
            pattern_display = html.Div([
                html.P("No patterns detected"),
                html.P("Collecting more data...")
            ])
        
        # Create anomaly detection display
        if not anomalies.empty:
            anomaly_display = html.Div([
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th('Date'),
                            html.Th('Price'),
                            html.Th('Volume'),
                            html.Th('Volatility')
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(idx.strftime('%Y-%m-%d %H:%M')),
                            html.Td(f"${row['close']:.2f}"),
                            html.Td(f"{row['volume']:.0f}"),
                            html.Td(f"{row['volatility']*100:.2f}%")
                        ]) for idx, row in anomalies.tail(5).iterrows()  # Show last 5 anomalies
                    ])
                ])
            ])
        else:
            anomaly_display = html.Div([
                html.P("No anomalies detected"),
                html.P("Collecting more data...")
            ])
        
        # Create main chart with candlesticks and moving averages
        fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03,
                                row_heights=[0.7, 0.3])
        
        # Add candlestick chart
        fig_main.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add predicted price
        if price_prediction:
            fig_main.add_trace(
                go.Scatter(
                    x=[df.index[-1]],
                    y=[price_prediction['predicted_price']],
                    mode='markers',
                    name='Predicted Price',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color='yellow',
                        line=dict(width=2, color='black')
                    )
                ),
                row=1, col=1
            )
        
        # Add anomalies
        if not anomalies.empty:
            fig_main.add_trace(
                go.Scatter(
                    x=anomalies.index,
                    y=anomalies['close'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        symbol='x',
                        size=10,
                        color='red',
                        line=dict(width=2, color='black')
                    )
                ),
                row=1, col=1
            )
        
        # Add patterns
        if patterns:
            for pattern in patterns[-5:]:  # Show last 5 patterns
                fig_main.add_trace(
                    go.Scatter(
                        x=[pattern['date']],
                        y=[pattern['price']],
                        mode='markers+text',
                        name=pattern['type'],
                        text=[pattern['type']],
                        textposition='top center',
                        marker=dict(
                            symbol='diamond',
                            size=12,
                            color='purple',
                            line=dict(width=2, color='white')
                        )
                    ),
                    row=1, col=1
                )
        
        # Add entry points
        entry_points = df[df['Entry']]
        fig_main.add_trace(
            go.Scatter(
                x=entry_points.index,
                y=entry_points['close'],
                mode='markers+text',
                name='Entry (BUY)',
                text=['BUY'] * len(entry_points),
                textposition='top center',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                    line=dict(width=2, color='white')
                )
            ),
            row=1, col=1
        )
        
        # Add exit points
        exit_points = df[df['Exit']]
        fig_main.add_trace(
            go.Scatter(
                x=exit_points.index,
                y=exit_points['close'],
                mode='markers+text',
                name='Exit (SELL)',
                text=['SELL'] * len(exit_points),
                textposition='bottom center',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red',
                    line=dict(width=2, color='white')
                )
            ),
            row=1, col=1
        )
        
        # Add moving averages
        fig_main.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        fig_main.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add Bollinger Bands
        fig_main.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        fig_main.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        
        # Add volume
        fig_main.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='rgba(0,0,255,0.3)'),
            row=2, col=1
        )
        
        # Update layout
        fig_main.update_layout(
            title=f'{pair} Price Chart ({timeframe})',
            yaxis_title='Price (USDT)',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        # Create indicators chart
        fig_indicators = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                     vertical_spacing=0.05,
                                     row_heights=[0.4, 0.3, 0.3])
        
        # RSI
        fig_indicators.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=1, col=1
        )
        fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        fig_indicators.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
            row=2, col=1
        )
        fig_indicators.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')),
            row=2, col=1
        )
        fig_indicators.add_trace(
            go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color='rgba(0,0,255,0.3)'),
            row=2, col=1
        )
        
        # Stochastic
        fig_indicators.add_trace(
            go.Scatter(x=df.index, y=df['Stoch_K'], name='Stoch %K', line=dict(color='blue')),
            row=3, col=1
        )
        fig_indicators.add_trace(
            go.Scatter(x=df.index, y=df['Stoch_D'], name='Stoch %D', line=dict(color='orange')),
            row=3, col=1
        )
        fig_indicators.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
        fig_indicators.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update indicators layout
        fig_indicators.update_layout(
            title='Technical Indicators',
            template='plotly_white'
        )
        
        # Generate trading signals
        signals = generate_trading_signals(df)
        latest = df.iloc[-1]
        
        # Create signals panel
        signals_panel = html.Div([
            html.H3('Trading Signals', style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.Div([
                html.Div([
                    html.H4('Current Price', style={'color': '#2c3e50'}),
                    html.P(f'${latest["close"]:.2f} USDT', style={'fontSize': '24px', 'color': '#27ae60'})
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4('RSI Signal', style={'color': '#2c3e50'}),
                    html.P(signals['RSI'], style={
                        'fontSize': '24px',
                        'color': 'green' if signals['RSI'] == 'Buy' else 'red' if signals['RSI'] == 'Sell' else 'gray'
                    })
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4('MACD Signal', style={'color': '#2c3e50'}),
                    html.P(signals['MACD'], style={
                        'fontSize': '24px',
                        'color': 'green' if signals['MACD'] == 'Buy' else 'red'
                    })
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4('Bollinger Bands', style={'color': '#2c3e50'}),
                    html.P(signals['BB'], style={
                        'fontSize': '24px',
                        'color': 'green' if signals['BB'] == 'Buy' else 'red' if signals['BB'] == 'Sell' else 'gray'
                    })
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4('Overall Signal', style={'color': '#2c3e50'}),
                    html.P(signals['Overall'], style={
                        'fontSize': '24px',
                        'color': 'green' if signals['Overall'] == 'Buy' else 'red' if signals['Overall'] == 'Sell' else 'gray'
                    })
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ])
        
        # Get recent trading points
        recent_points = get_recent_trading_points(df)
        
        # Create trading points panel
        trading_points_panel = html.Div([
            html.H3('Recent Trading Points', style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.Div([
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th('Date', style={'padding': '10px'}),
                            html.Th('Type', style={'padding': '10px'}),
                            html.Th('Action', style={'padding': '10px'}),
                            html.Th('Price', style={'padding': '10px'}),
                            html.Th('Reason', style={'padding': '10px'})
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(point['date'].strftime('%Y-%m-%d %H:%M'), style={'padding': '10px'}),
                            html.Td(point['type'], style={
                                'padding': '10px',
                                'color': 'green' if point['type'] == 'Entry' else 'red'
                            }),
                            html.Td(point['action'], style={
                                'padding': '10px',
                                'color': 'green' if point['action'] == 'BUY' else 'red',
                                'fontWeight': 'bold'
                            }),
                            html.Td(f'${point["price"]:.2f}', style={'padding': '10px'}),
                            html.Td(point['reason'], style={'padding': '10px'})
                        ]) for point in recent_points
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse'})
            ])
        ])
        
        # Calculate risk metrics
        if n_clicks_calc > 0 or n_clicks_update > 0:
            # Update risk manager settings
            risk_manager.update_account_balance(balance)
            risk_manager.risk_per_trade = risk_percent / 100
            
            # Get current price
            current_price = collector.get_current_price(pair)
            
            # Calculate stop loss and take profit prices
            sl_price = current_price * (1 - sl_percent/100)
            tp_price = current_price * (1 + tp_percent/100)
            
            # Calculate position size
            position = risk_manager.calculate_position_size(current_price, sl_price)
            
            # Calculate risk metrics
            historical_data = collector.get_historical_data(pair=pair, period='30d', interval='1h')
            risk_metrics = risk_manager.calculate_risk_metrics(
                historical_data,
                position['position_size'],
                sl_price,
                tp_price
            )
            
            # Create risk metrics display
            risk_metrics_display = html.Div([
                html.H4('Position and Risk Metrics', style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Div([
                    html.Div([
                        html.H5('Position Details', style={'color': '#2c3e50'}),
                        html.P(f"Position Size: {position['position_size']:.4f} {pair}"),
                        html.P(f"Position Value: ${position['position_value']:.2f}"),
                        html.P(f"Risk Amount: ${position['risk_amount']:.2f}"),
                        html.P(f"Risk Percentage: {position['risk_percentage']:.2f}%")
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.H5('Risk Metrics', style={'color': '#2c3e50'}),
                        html.P(f"Volatility: {risk_metrics['volatility']:.2f}%"),
                        html.P(f"Value at Risk (95%): ${risk_metrics['var_95']:.2f}"),
                        html.P(f"Maximum Loss: ${risk_metrics['max_loss']:.2f}"),
                        html.P(f"Maximum Profit: ${risk_metrics['max_profit']:.2f}")
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.H5('Trade Parameters', style={'color': '#2c3e50'}),
                        html.P(f"Entry Price: ${current_price:.2f}"),
                        html.P(f"Stop Loss: ${sl_price:.2f}"),
                        html.P(f"Take Profit: ${tp_price:.2f}"),
                        html.P(f"Risk-Reward Ratio: {risk_metrics['risk_reward_ratio']:.2f}")
                    ], style={'width': '30%', 'display': 'inline-block'})
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ])
        else:
            risk_metrics_display = html.Div("Enter your risk parameters and click 'Calculate Position Size'")
        
        return (fig_main, fig_indicators, signals_panel, trading_points_panel, risk_metrics_display,
                price_prediction_display, market_regime_display, pattern_display, anomaly_display)
        
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}", exc_info=True)
        return {}, {}, f"Error: {str(e)}", f"Error: {str(e)}", html.Div(f"Error: {str(e)}"), 'Error', 'Error', 'Error', 'Error'

# Create server variable for deployment
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8050))) 