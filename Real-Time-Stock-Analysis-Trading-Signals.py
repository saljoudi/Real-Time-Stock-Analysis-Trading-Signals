import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
from tqdm import tqdm
import plotly.graph_objects as go
import ta

# Global variables for progress tracking
progress = 0
total_combinations = 0

# --- Function to simulate the strategy without delay ---
def process_params(params, df, ticker, initial_investment=100000):
    """
    Simulate the trading strategy for a given set of technical parameters.
    
    Parameters:
        params (tuple): (sma_short, sma_long, rsi_threshold, adl_short, adl_long)
        df (DataFrame): Data with columns 'Close', 'High', 'Low', 'Volume', etc.
        ticker (str): Stock ticker.
        initial_investment (float): Starting portfolio value.
    
    Returns:
        dict or None: Contains performance metrics, trade details, and parameter set.
    """
    sma_short, sma_long, rsi_threshold, adl_short, adl_long = params

    if len(df) < max(sma_short, sma_long, adl_short, adl_long):
        return None

    # Compute rolling averages
    df['SMA_Short'] = df['Close'].rolling(window=sma_short).mean()
    df['SMA_Long'] = df['Close'].rolling(window=sma_long).mean()
    df['ADL_Short_SMA'] = df['ADL'].rolling(window=adl_short).mean()
    df['ADL_Long_SMA'] = df['ADL'].rolling(window=adl_long).mean()

    # Generate trading signals
    df['Signal'] = df.apply(
        lambda row: -1 if (
            row['Close'] >= row['SMA_Short'] and
            row['SMA_Short'] > row['SMA_Long'] and
            row['ADL_Short_SMA'] > row['ADL_Long_SMA'] and
            row['RSI'] >= rsi_threshold and
            row['MACD'] > row['MACD_Signal']
        ) else (
            1 if (
                row['Close'] < row['SMA_Short'] and
                row['SMA_Short'] < row['SMA_Long']
            ) else 0
        ),
        axis=1
    )

    position = 0
    cash = initial_investment
    portfolio_values = []
    buy_price = None
    trade_start = None
    trades = []
    number_of_trades = 0
    amount_invested = 0

    for index, row in df.iterrows():
        if position == 0:
            if row['Signal'] == 1:
                buy_price = row['Close']
                position = cash / buy_price
                amount_invested = cash
                cash = 0
                trade_start = index  # record buy time
                number_of_trades += 1
        elif position > 0:
            if row['Signal'] == -1:
                sell_price = row['Close']
                cash = position * sell_price
                profit = cash - amount_invested  # Corrected calculation
                profit_percentage = ((sell_price - buy_price) / buy_price) * 100
                minutes_held = int((index - trade_start).total_seconds() / 60)
                trades.append({
                    "Buy Time": trade_start,
                    "Sell Time": index,
                    "Buy Price": buy_price,
                    "Sell Price": sell_price,
                    "Minutes Held": minutes_held,
                    "Amount Invested": amount_invested,
                    "Profit": profit,
                    "Profit Percentage": profit_percentage
                })
                position = 0
                buy_price = None
                amount_invested = 0

        portfolio_value = position * row['Close'] if position > 0 else cash
        portfolio_values.append(portfolio_value)

    if not portfolio_values:
        return None

    final_value = portfolio_values[-1]
    total_return = final_value - initial_investment
    percentage_return = (total_return / initial_investment) * 100

    # Maximum drawdown
    port_array = np.array(portfolio_values)
    roll_max = np.maximum.accumulate(port_array)
    daily_drawdown = port_array / roll_max - 1.0
    max_drawdown = daily_drawdown.min() * 100 if daily_drawdown.size else 0

    daily_returns = pd.Series(portfolio_values).pct_change().fillna(0)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

    profitable_trades = [t for t in trades if t['Profit'] > 0]
    win_rate = len(profitable_trades) / number_of_trades if number_of_trades > 0 else 0
    total_profit = sum(t['Profit'] for t in trades if t['Profit'] > 0)
    total_loss = -sum(t['Profit'] for t in trades if t['Profit'] < 0)
    profit_factor = (total_profit / total_loss) if total_loss > 0 else np.inf

    return {
        'params': params,
        'final_value': final_value,
        'percentage_return': percentage_return,
        'number_of_trades': number_of_trades,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades': trades,
        'portfolio_values': portfolio_values
    }

# Initialize the Dash app with the Lux theme for a professional look.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# Define the app layout using Bootstrap components.
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Trading Strategy Analyzer", className="text-center mb-4 mt-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Input Parameters", className="card-title"),
                    dbc.Label("Ticker:"),
                    dcc.Input(id='ticker-input', type='text', value='AAPL', className="mb-3", style={'width': '100%'}),
                    dbc.Label("Days (1-5):"),
                    dcc.Input(id='days-input', type='number', value=1, min=1, max=5, step=1, className="mb-3", style={'width': '100%'}),
                    dbc.Label("Interval:"),
                    dcc.Dropdown(
                        id='interval-dropdown',
                        options=[
                            {'label': '1 Minute', 'value': '1m'},
                            {'label': '2 Minutes', 'value': '2m'},
                            {'label': '5 Minutes', 'value': '5m'}
                        ],
                        value='1m',
                        className="mb-3",
                        style={'width': '100%'}
                    ),
                    dbc.Button("Analyze", id="analyze-button", color="primary", className="mt-3", style={'width': '100%'})
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Progress", className="card-title"),
                    dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, color="success", className="mb-3"),
                    html.Div(id='progress-text', className='text-center'),
                    html.H4("Strategy Summary", className="card-title"),
                    html.Pre(id='summary-output', style={'whiteSpace': 'pre-wrap', 'font-family': 'monospace'})
                ])
            ])
        ], width=9)
    ]),
    # Price Chart on top
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4("Price Chart", className="card-title"),
                    dcc.Graph(id='price-chart')
                ])
            ]),
            width=12
        )
    ], className="mb-4"),
    # Portfolio Chart below
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4("Portfolio Value Chart", className="card-title"),
                    dcc.Graph(id='portfolio-chart')
                ])
            ]),
            width=12
        )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Trade Details", className="card-title"),
                    html.Div(id='trades-table')
                ])
            ])
        ], width=12)
    ]),
    # Existing interval for progress updates (every 500 ms)
    dcc.Interval(id='progress-interval', interval=500, n_intervals=0),
    # New interval for auto-updating analysis every 5 minutes (300,000 ms)
    dcc.Interval(id='update-interval', interval=300000, n_intervals=0)
], fluid=True)

# Callback to update progress bar and text from global variables.
@app.callback(
    [Output('progress-bar', 'value'),
     Output('progress-text', 'children')],
    [Input('progress-interval', 'n_intervals')]
)
def update_progress(n_intervals):
    global progress, total_combinations
    if total_combinations:
        prog_percent = int((progress / total_combinations) * 100)
        return prog_percent, f"Progress: {prog_percent}% ({progress}/{total_combinations})"
    return 0, ""

# Callback for performing grid search analysis.
# This callback now triggers from both the "Analyze" button and the update interval.
@app.callback(
    [Output('summary-output', 'children'),
     Output('trades-table', 'children'),
     Output('price-chart', 'figure'),
     Output('portfolio-chart', 'figure')],
    [Input('analyze-button', 'n_clicks'),
     Input('update-interval', 'n_intervals')],
    [State('ticker-input', 'value'),
     State('days-input', 'value'),
     State('interval-dropdown', 'value')]
)
def perform_analysis(n_clicks, n_intervals, ticker, days, interval):
    # If neither the Analyze button nor the update interval has triggered an update, return empty figures.
    if (n_clicks is None or n_clicks == 0) and n_intervals == 0:
        return "", "", go.Figure(), go.Figure()
    
    # Download intraday data for the specified number of days.
    period_str = f"{days}d"
    df = yf.download(ticker, period=period_str, interval=interval, progress=False)
    if df.empty:
        return f"No data found for ticker {ticker}.", "", go.Figure(), go.Figure()
    
    # Preprocess data: ensure datetime index and fill missing Volume.
    df.index = pd.to_datetime(df.index)
    df['Date'] = df.index.date
    daily_avg = df.groupby('Date')['Volume'].transform('mean')
    df['Volume'] = df['Volume'].replace(0, pd.NA).fillna(daily_avg)
    df = df.drop(columns=['Date'])
    
    # Compute technical indicators.
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd_obj = ta.trend.MACD(df['Close'])
    df['MACD'] = macd_obj.macd()
    df['MACD_Signal'] = macd_obj.macd_signal()
    df['ADL'] = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
    
    # Define grid ranges.
    sma_short_range = range(10, 20, 10)
    sma_long_range = range(10, 50, 10)
    rsi_threshold_range = range(45, 55, 10)
    adl_short_range = range(10, 20, 10)
    adl_long_range = range(10, 50, 10)
    parameter_grid = list(product(sma_short_range, sma_long_range, rsi_threshold_range, adl_short_range, adl_long_range))
    total_combinations = len(parameter_grid)
    
    best_result = None
    best_perf = float('-inf')
    progress = 0  # Reset progress
    
    # Loop over each parameter combination and update progress.
    for params in tqdm(parameter_grid, desc="Analyzing parameters", leave=False):
        result = process_params(params, df.copy(), ticker)
        progress += 1
        if result is not None and result['percentage_return'] > best_perf:
            best_perf = result['percentage_return']
            best_result = result

    if best_result is None:
        return "No valid simulation results found.", "", go.Figure(), go.Figure()
    
    # Prepare strategy summary text.
    sma_short, sma_long, rsi_threshold, adl_short, adl_long = best_result['params']
    summary_text = (
        f"Best Parameters:\n"
        f"  SMA Short: {sma_short}\n"
        f"  SMA Long: {sma_long}\n"
        f"  RSI Threshold: {rsi_threshold}\n"
        f"  ADL Short SMA: {adl_short}\n"
        f"  ADL Long SMA: {adl_long}\n\n"
        f"Initial Investment: 100,000.00\n"
        f"Final Portfolio Value: {best_result['final_value']:,.2f}\n"
        f"Total Return: {best_result['final_value'] - 100000:,.2f}\n"
        f"Percentage Return: {best_result['percentage_return']:.2f}%\n"
        f"Number of Trades: {best_result['number_of_trades']}\n"
        f"Max Drawdown: {best_result['max_drawdown']:.2f}%\n"
        f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}\n"
        f"Win Rate: {best_result['win_rate']*100:.2f}%\n"
        f"Profit Factor: {best_result['profit_factor']:.2f}"
    )
    
    # Build the price chart.
    # Recompute signals for plotting using the best parameters.
    df['SMA_Short'] = df['Close'].rolling(window=sma_short).mean()
    df['SMA_Long'] = df['Close'].rolling(window=sma_long).mean()
    df['ADL_Short_SMA'] = df['ADL'].rolling(window=adl_short).mean()
    df['ADL_Long_SMA'] = df['ADL'].rolling(window=adl_long).mean()
    df['Signal'] = df.apply(
        lambda row: -1 if (
            row['Close'] >= row['SMA_Short'] and
            row['SMA_Short'] > row['SMA_Long'] and
            row['ADL_Short_SMA'] > row['ADL_Long_SMA'] and
            row['RSI'] >= rsi_threshold and
            row['MACD'] > row['MACD_Signal']
        ) else (
            1 if (
                row['Close'] < row['SMA_Short'] and
                row['SMA_Short'] < row['SMA_Long']
            ) else 0
        ),
        axis=1
    )
    df['Buy Signal'] = df['Signal'] == 1
    df['Sell Signal'] = df['Signal'] == -1

    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=df.index, y=df['Close'],
                                   mode='lines', name='Close Price', line=dict(color='blue')))
    price_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Short'],
                                   mode='lines', name='SMA Short', line=dict(color='orange')))
    price_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Long'],
                                   mode='lines', name='SMA Long', line=dict(color='green')))
    buy_df = df[df['Buy Signal']]
    sell_df = df[df['Sell Signal']]
    price_fig.add_trace(go.Scatter(
        x=buy_df.index, y=buy_df['Close'],
        mode='markers', marker=dict(symbol='triangle-up', color='green', size=10),
        name='Buy Signal'
    ))
    price_fig.add_trace(go.Scatter(
        x=sell_df.index, y=sell_df['Close'],
        mode='markers', marker=dict(symbol='triangle-down', color='red', size=10),
        name='Sell Signal'
    ))
    price_fig.update_layout(
        title=f"{ticker.upper()} - Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        template='plotly_white'
    )
    
    # Build the portfolio value chart.
    portfolio_fig = go.Figure()
    portfolio_fig.add_trace(go.Scatter(
        x=list(range(len(best_result['portfolio_values']))),
        y=best_result['portfolio_values'],
        mode='lines', name='Portfolio Value', line=dict(color='purple')
    ))
    portfolio_fig.update_layout(
        title=f"{ticker.upper()} - Portfolio Value Over Trades",
        xaxis_title="Trade Number",
        yaxis_title="Portfolio Value",
        template='plotly_white'
    )
    
    # Format trade details for professional display.
    formatted_trades = []
    for trade in best_result['trades']:
        buy_date = trade["Buy Time"].strftime("%Y-%m-%d") if hasattr(trade["Buy Time"], "strftime") else str(trade["Buy Time"])
        formatted_trade = {
            "Date": buy_date,  # New date column as the first column
            "Buy Time": trade["Buy Time"].strftime("%H:%M:%S") if hasattr(trade["Buy Time"], "strftime") else str(trade["Buy Time"]),
            "Sell Time": trade["Sell Time"].strftime("%H:%M:%S") if hasattr(trade["Sell Time"], "strftime") else str(trade["Sell Time"]),
            "Buy Price": f"{trade['Buy Price']:.2f}",
            "Sell Price": f"{trade['Sell Price']:.2f}",
            "Minutes Held": trade["Minutes Held"],
            "Amount Invested": f"{trade['Amount Invested']:.2f}",
            "Profit": f"{trade['Profit']:.2f}",
            "Profit Percentage": f"{trade['Profit Percentage']:.2f}%"
        }
        formatted_trades.append(formatted_trade)
    
    # Create a Bootstrap table from the formatted trades.
    trades_df = pd.DataFrame(formatted_trades)
    trades_table = dbc.Table.from_dataframe(trades_df, striped=True, bordered=True, hover=True, responsive=True)
    
    return summary_text, trades_table, price_fig, portfolio_fig

if __name__ == '__main__':
    app.run_server(debug=True) 
