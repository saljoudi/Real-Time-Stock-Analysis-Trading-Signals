import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objs as go

from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# For optimization parallel processing
from joblib import Parallel, delayed
from itertools import product

#####################################
# 1) REAL-TIME TRADING ANALYSIS
#####################################
def process_stock(df, stock_symbol, sma_short, sma_long, rsi_threshold, adl_short, adl_long, initial_investment=100000):
    # ------------------ 1) Moving Averages ------------------
    df['SMA_Short'] = df['Close'].rolling(window=int(sma_short)).mean()
    df['SMA_Long']  = df['Close'].rolling(window=int(sma_long)).mean()

    # ------------------ 2) RSI ------------------
    rsi_series = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['RSI'] = pd.Series(rsi_series.values, index=df.index)

    # ------------------ 3) MACD ------------------
    macd = ta.trend.MACD(df['Close'])
    df['MACD']        = macd.macd().squeeze()
    df['MACD_Signal'] = macd.macd_signal().squeeze()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # ------------------ 4) ADL ------------------
    adl_series = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
    df['ADL'] = pd.Series(adl_series.values, index=df.index)
    df['ADL_Short_SMA'] = df['ADL'].rolling(window=int(adl_short)).mean()
    df['ADL_Long_SMA']  = df['ADL'].rolling(window=int(adl_long)).mean()

    # ------------------ 5) Trading Signals ------------------
    df['Signal'] = df.apply(
        lambda row: -1 if (
            row['Close'] >= row['SMA_Short'] and
            row['SMA_Short'] > row['SMA_Long']   and
            row['ADL_Short_SMA'] > row['ADL_Long_SMA'] and
            row['RSI'] >= int(rsi_threshold) and
            row['MACD'] > row['MACD_Signal']
        ) else (
            1 if (
                row['Close'] < row['SMA_Short'] and
                row['SMA_Short'] < row['SMA_Long']
            ) else 0
        ),
        axis=1
    )

    # ------------------ 6) Backtest / Portfolio Logic ------------------
    position = 0
    cash = initial_investment
    portfolio_value = initial_investment
    buy_price = None
    trade_start = None
    trades = []
    number_of_trades = 0
    portfolio_values = []
    amount_invested = 0

    for index, row in df.iterrows():
        if row['Signal'] == 1 and position == 0:
            # Buy signal
            buy_price = row['Close']
            position = cash / buy_price
            amount_invested = cash
            cash = 0
            trade_start = index
            number_of_trades += 1

        elif row['Signal'] == -1 and position > 0:
            # Sell signal
            sell_price = row['Close']
            cash = position * sell_price
            profit = cash - amount_invested
            profit_percentage = ((sell_price - buy_price) / buy_price) * 100
            minutes_held = (index - trade_start).total_seconds() / 60.0

            trades.append({
                'Buy Time': trade_start,
                'Sell Time': index,
                'Buy Price': buy_price,
                'Sell Price': sell_price,
                'Minutes Held': minutes_held,
                'Amount Invested': amount_invested,
                'Profit': profit,
                'Profit Percentage': profit_percentage
            })

            position = 0
            buy_price = None
            amount_invested = 0

        portfolio_value = position * row['Close'] if position > 0 else cash
        portfolio_values.append(portfolio_value)

    final_value = portfolio_values[-1] if portfolio_values else initial_investment
    total_return = final_value - initial_investment
    percentage_return = (total_return / initial_investment) * 100

    daily_returns = pd.Series(portfolio_values).pct_change().fillna(0)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0

    profitable_trades = [t for t in trades if t['Profit'] > 0]
    win_rate = len(profitable_trades) / number_of_trades if number_of_trades > 0 else 0

    avg_minutes_held = sum([t['Minutes Held'] for t in trades]) / number_of_trades if number_of_trades > 0 else 0

    df['Buy Signal']  = df['Signal'] == 1
    df['Sell Signal'] = df['Signal'] == -1

    return {
        'final_value': final_value,
        'percentage_return': percentage_return,
        'number_of_trades': number_of_trades,
        'average_minutes_held': avg_minutes_held,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'trades': trades,
        'portfolio_values': portfolio_values,
        'df': df
    }

#####################################
# 2) PARAMETER OPTIMIZATION FUNCTION
#####################################
def process_params(params, df, ticker, initial_investment=100000):
    """
    Evaluate one set of parameters without delay.
    params: (sma_short, sma_long, rsi_threshold, adl_short, adl_long)
    Assumes df already has RSI, MACD, MACD_Signal, and ADL computed.
    """
    sma_short, sma_long, rsi_threshold, adl_short, adl_long = params

    # Ensure sufficient data length
    if len(df) < max(sma_short, sma_long, adl_short, adl_long):
        return None

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

    position = 0
    cash = initial_investment
    portfolio_value = initial_investment
    buy_price = None
    trade_start = None
    trades = []
    number_of_trades = 0
    portfolio_values = []
    amount_invested = 0

    for index, row in df.iterrows():
        if position == 0 and row['Signal'] == 1:
            buy_price = row['Close']
            position = cash / buy_price
            amount_invested = cash
            cash = 0
            trade_start = index
            number_of_trades += 1
        elif position > 0 and row['Signal'] == -1:
            sell_price = row['Close']
            cash = position * sell_price
            profit = cash - amount_invested
            profit_percentage = ((sell_price - buy_price) / buy_price) * 100
            days_held = (index - trade_start).days
            trades.append({
                'Ticker': ticker,
                'Sell Date': index,
                'Buy Price': buy_price,
                'Sell Price': sell_price,
                'Days Held': days_held,
                'Amount Invested': amount_invested,
                'Profit': profit,
                'Profit Percentage': profit_percentage
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

    portfolio_values_array = np.array(portfolio_values)
    roll_max = np.maximum.accumulate(portfolio_values_array)
    daily_drawdown = portfolio_values_array / roll_max - 1.0
    max_drawdown = daily_drawdown.min() * 100 if daily_drawdown.size else 0

    daily_returns = pd.Series(portfolio_values_array).pct_change().fillna(0)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

    profitable_trades = [t for t in trades if t['Profit'] > 0]
    win_rate = len(profitable_trades) / number_of_trades if number_of_trades > 0 else 0
    total_profit = sum(t['Profit'] for t in trades if t['Profit'] > 0)
    total_loss = -sum(t['Profit'] for t in trades if t['Profit'] < 0)
    profit_factor = (total_profit / total_loss) if total_loss > 0 else np.inf
    average_days_held = sum(t['Days Held'] for t in trades) / number_of_trades if number_of_trades > 0 else 0

    return {
        'params': params,
        'final_value': final_value,
        'percentage_return': percentage_return,
        'number_of_trades': number_of_trades,
        'average_days_held': average_days_held,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades': trades,
        'portfolio_values': portfolio_values
    }

#####################################
# DASH APP LAYOUT
#####################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# Layout for Trading Analysis tab (left: inputs, right: analysis output)
trading_analysis_layout = dbc.Row([
    dbc.Col(
        dbc.Card([
            dbc.CardHeader("Trading Analysis Input Parameters"),
            dbc.CardBody([
                html.Div([
                    dbc.Label("Stock Symbol:", html_for='stock-symbol'),
                    dbc.Input(id='stock-symbol', type='text', value='AAPL', debounce=True),
                    dbc.FormText("e.g., AAPL or for Saudi tickers use numeric value then '.SR'")
                ], className="mb-3"),
                html.Div([
                    dbc.Label("Time Period (Days):", html_for='time-period'),
                    dcc.Dropdown(
                        id='time-period',
                        options=[
                            {'label': '1 Day', 'value': '1d'},
                            {'label': '2 Days', 'value': '2d'},
                            {'label': '3 Days', 'value': '3d'},
                            {'label': '4 Days', 'value': '4d'},
                            {'label': '5 Days', 'value': '5d'},
                        ],
                        value='1d'
                    )
                ], className="mb-3"),
                html.Div([
                    dbc.Label("SMA Short:", html_for='sma-short'),
                    dbc.Input(id='sma-short', type='number', value=7, min=1, debounce=True)
                ], className="mb-3"),
                html.Div([
                    dbc.Label("SMA Long:", html_for='sma-long'),
                    dbc.Input(id='sma-long', type='number', value=10, min=1, debounce=True)
                ], className="mb-3"),
                html.Div([
                    dbc.Label("RSI Threshold:", html_for='rsi-threshold'),
                    dbc.Input(id='rsi-threshold', type='number', value=40, min=0, max=100, debounce=True)
                ], className="mb-3"),
                html.Div([
                    dbc.Label("ADL Short:", html_for='adl-short'),
                    dbc.Input(id='adl-short', type='number', value=13, min=1, debounce=True)
                ], className="mb-3"),
                html.Div([
                    dbc.Label("ADL Long:", html_for='adl-long'),
                    dbc.Input(id='adl-long', type='number', value=30, min=1, debounce=True)
                ], className="mb-3"),
                dbc.Button('Analyze Stock', id='submit-button', color='primary', className='w-100')
            ])
        ], className="mb-4"),
        width=3
    ),
    dbc.Col([
        dbc.Tabs([
            dbc.Tab(label="Price Chart", tab_id="tab-price-chart", children=[
                dbc.CardBody(dcc.Graph(id='stock-graph', style={'height': '70vh'}))
            ]),
            dbc.Tab(label="MACD", tab_id="tab-macd", children=[
                dbc.CardBody(dcc.Graph(id='macd-graph', style={'height': '70vh'}))
            ]),
            dbc.Tab(label="RSI", tab_id="tab-rsi", children=[
                dbc.CardBody(dcc.Graph(id='rsi-graph', style={'height': '70vh'}))
            ]),
            dbc.Tab(label="Performance Metrics", tab_id="tab-metrics", children=[
                dbc.CardBody(html.Div(id='performance-metrics', className="mt-3"))
            ]),
            dbc.Tab(label="Trade Details", tab_id="tab-trades", children=[
                dbc.CardBody(html.Div(id='trade-details', className="mt-3"))
            ]),
        ], id='output-tabs', active_tab='tab-price-chart', className="mt-3")
    ], width=9)
], className="mb-5")

# Layout for Parameter Optimization tab
optimization_layout = dbc.Row([
    dbc.Col(
        dbc.Card([
            dbc.CardHeader("Optimization Input Parameters"),
            dbc.CardBody([
                html.Div([
                    dbc.Label("Stock Symbol:", html_for='opt-symbol'),
                    dbc.Input(id='opt-symbol', type='text', value='AAPL', debounce=True),
                    dbc.FormText("e.g., AAPL or append '.SR' for Saudi tickers")
                ], className="mb-3"),
                html.Div([
                    dbc.Label("Time Period (Days):", html_for='opt-time-period'),
                    dcc.Dropdown(
                        id='opt-time-period',
                        options=[
                            {'label': '1 Day', 'value': '1d'},
                            {'label': '2 Days', 'value': '2d'},
                            {'label': '3 Days', 'value': '3d'},
                            {'label': '4 Days', 'value': '4d'},
                            {'label': '5 Days', 'value': '5d'},
                        ],
                        value='1d'
                    )
                ], className="mb-3"),
                dbc.Button('Optimize Parameters', id='optimize-button', color='warning', className='w-100')
            ])
        ], className="mb-4"),
        width=3
    ),
    dbc.Col(
        dbc.Card([
            dbc.CardHeader("Optimization Results"),
            dbc.CardBody([
                html.Div(id='optimization-results')
            ])
        ]),
        width=9
    )
], className="mb-5")

# Main layout with two tabs
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.H1("Advanced Stock Analysis & Optimization App", className="text-center text-primary mb-4"), width=12)
    ]),
    dbc.Tabs([
        dbc.Tab(label="Trading Analysis", tab_id="trading-analysis", children=[trading_analysis_layout]),
        dbc.Tab(label="Parameter Optimization", tab_id="optimization", children=[optimization_layout])
    ], id='main-tabs', active_tab='trading-analysis'),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
])

#####################################
# CALLBACKS
#####################################
# Callback for Trading Analysis tab
@app.callback(
    [
        Output('stock-graph', 'figure'),
        Output('macd-graph', 'figure'),
        Output('rsi-graph', 'figure'),
        Output('performance-metrics', 'children'),
        Output('trade-details', 'children')
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('submit-button', 'n_clicks')
    ],
    [
        State('stock-symbol', 'value'),
        State('time-period', 'value'),
        State('sma-short', 'value'),
        State('sma-long', 'value'),
        State('rsi-threshold', 'value'),
        State('adl-short', 'value'),
        State('adl-long', 'value')
    ]
)
def update_analysis(n_intervals, n_clicks, stock_symbol, time_period,
                    sma_short, sma_long, rsi_threshold, adl_short, adl_long):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, "", ""
    try:
        if stock_symbol.isdigit():
            stock_symbol += '.SR'
        df = yf.download(stock_symbol, period=time_period, interval='1m')
        df.index = pd.to_datetime(df.index)
        if df.empty:
            msg = f"No data found for {stock_symbol} over the last {time_period}."
            return dash.no_update, dash.no_update, dash.no_update, html.P(msg, className="text-danger"), ""
        result = process_stock(df.copy(), stock_symbol, sma_short, sma_long, rsi_threshold, adl_short, adl_long)
        df = result['df']

        # Price Chart
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines', name='Close Price', line=dict(color='blue', width=2)
        ))
        price_fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_Short'],
            mode='lines', name=f'SMA Short ({sma_short})', line=dict(color='orange', dash='dash', width=1)
        ))
        price_fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_Long'],
            mode='lines', name=f'SMA Long ({sma_long})', line=dict(color='green', dash='dot', width=1)
        ))
        # Buy/Sell markers
        buy_signals = df[df['Buy Signal']]
        sell_signals = df[df['Sell Signal']]
        price_fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['Close'],
            mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=10)
        ))
        price_fig.add_trace(go.Scatter(
            x=sell_signals.index, y=sell_signals['Close'],
            mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=10)
        ))
        price_fig.update_layout(
            title={'text': f"Price with Buy/Sell Signals - {stock_symbol}", 'x':0.5},
            xaxis_title='Date/Time', yaxis_title='Price', template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=80, b=40)
        )

        # MACD Chart
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')
        ))
        macd_fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='orange')
        ))
        macd_fig.add_trace(go.Bar(
            x=df.index, y=df['MACD_Histogram'], name='MACD Histogram', marker_color='gray', opacity=0.5
        ))
        macd_fig.update_layout(
            title={'text': f"MACD - {stock_symbol}", 'x':0.5},
            xaxis_title='Date/Time', yaxis_title='MACD', template='plotly_white',
            margin=dict(l=40, r=40, t=80, b=40), showlegend=True
        )

        # RSI Chart
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')
        ))
        rsi_fig.add_shape(type='line', xref='paper', x0=0, x1=1, y0=70, y1=70,
                           line=dict(color='red', dash='dash'))
        rsi_fig.add_shape(type='line', xref='paper', x0=0, x1=1, y0=30, y1=30,
                           line=dict(color='green', dash='dash'))
        rsi_fig.update_layout(
            title={'text': f"RSI - {stock_symbol}", 'x':0.5},
            xaxis_title='Date/Time', yaxis_title='RSI', template='plotly_white',
            margin=dict(l=40, r=40, t=80, b=40), yaxis=dict(range=[0,100])
        )

        # Performance Metrics
        trades = result['trades']
        perf_metrics = [
            html.H5("Performance Metrics"),
            html.Ul([
                html.Li("Initial Investment: 100,000 SAR"),
                html.Li(f"Final Portfolio Value: {result['final_value']:,.2f} SAR"),
                html.Li(f"Total Return: {result['final_value'] - 100000:,.2f} SAR"),
                html.Li(f"Percentage Return: {result['percentage_return']:.2f}%"),
                html.Li(f"Number of Trades: {result['number_of_trades']}"),
                html.Li(f"Average Minutes Held per Trade: {result['average_minutes_held']:.2f}"),
                html.Li(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}"),
                html.Li(f"Win Rate: {result['win_rate']*100:.2f}%"),
            ], className="list-unstyled")
        ]

        # Trade Details
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['Buy Price'] = trades_df['Buy Price'].map('{:,.2f} SAR'.format)
            trades_df['Sell Price'] = trades_df['Sell Price'].map('{:,.2f} SAR'.format)
            trades_df['Amount Invested'] = trades_df['Amount Invested'].map('{:,.2f} SAR'.format)
            trades_df['Profit'] = trades_df['Profit'].map('{:,.2f} SAR'.format)
            trades_df['Profit Percentage'] = trades_df['Profit Percentage'].map('{:.2f}%'.format)
            trades_df['Minutes Held'] = trades_df['Minutes Held'].map('{:.2f}'.format)
            trades_df['Buy Time'] = trades_df['Buy Time'].astype(str)
            trades_df['Sell Time'] = trades_df['Sell Time'].astype(str)
            trade_details = dbc.Table.from_dataframe(trades_df, striped=True, bordered=True, hover=True, responsive=True)
            trade_details_output = [html.H5("Trade Details"), trade_details]
        else:
            trade_details_output = [html.P("No trades were made with the given parameters.")]
        
        return price_fig, macd_fig, rsi_fig, perf_metrics, trade_details_output

    except Exception as e:
        err_msg = html.P(f"An error occurred: {str(e)}", className="text-danger mt-3")
        return dash.no_update, dash.no_update, dash.no_update, err_msg, ""

# Callback for Parameter Optimization tab
@app.callback(
    Output('optimization-results', 'children'),
    Input('optimize-button', 'n_clicks'),
    [
        State('opt-symbol', 'value'),
        State('opt-time-period', 'value')
    ]
)
def optimize_parameters(n_clicks, opt_symbol, opt_time_period):
    if not n_clicks:
        return ""
    try:
        ticker = opt_symbol
        if ticker.isdigit():
            ticker += '.SR'
        # Download data
        df = yf.download(ticker, period=opt_time_period, interval='1m')
        df.index = pd.to_datetime(df.index)
        if df.empty:
            return html.P(f"No data found for {ticker} over the last {opt_time_period}.", className="text-danger")
        # Precompute technical indicators (RSI, MACD, ADL)
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['ADL'] = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
        
        # Define parameter ranges
        sma_short_range = list(range(5, 20, 2))
        sma_long_range = list(range(10, 50, 5))
        rsi_threshold_range = list(range(40, 55, 5))
        adl_short_range = list(range(5, 20, 2))
        adl_long_range = list(range(10, 50, 5))
        param_grid = list(product(sma_short_range, sma_long_range, rsi_threshold_range, adl_short_range, adl_long_range))
        
        # Run optimization in parallel (using a copy of df for each evaluation)
        tech_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(process_params)(params, df.copy(), ticker)
            for params in param_grid
        )
        tech_results = [res for res in tech_results if res is not None]
        if not tech_results:
            return html.P("No valid parameter results found.", className="text-danger")
        best_result = max(tech_results, key=lambda x: x['percentage_return'])
        best_params = best_result['params']
        
        # Create a line chart of portfolio value over time from the best result
        portfolio_vals = best_result['portfolio_values']
        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            x=list(range(len(portfolio_vals))),
            y=portfolio_vals,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='purple')
        ))
        portfolio_fig.update_layout(
            title="Portfolio Value Over Time (Optimized Strategy)",
            xaxis_title="Trade Number",
            yaxis_title="Portfolio Value",
            template='plotly_white'
        )
        
        results_layout = html.Div([
            html.H5("Best Parameters Found:"),
            html.Ul([
                html.Li(f"SMA Short: {best_params[0]}"),
                html.Li(f"SMA Long: {best_params[1]}"),
                html.Li(f"RSI Threshold: {best_params[2]}"),
                html.Li(f"ADL Short: {best_params[3]}"),
                html.Li(f"ADL Long: {best_params[4]}")
            ]),
            html.H5("Performance Metrics:"),
            html.Ul([
                html.Li(f"Final Portfolio Value: {best_result['final_value']:,.2f} SAR"),
                html.Li(f"Percentage Return: {best_result['percentage_return']:.2f}%"),
                html.Li(f"Number of Trades: {best_result['number_of_trades']}"),
                html.Li(f"Average Days Held: {best_result['average_days_held']:.2f}"),
                html.Li(f"Max Drawdown: {best_result['max_drawdown']:.2f}%"),
                html.Li(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}"),
                html.Li(f"Win Rate: {best_result['win_rate']*100:.2f}%"),
                html.Li(f"Profit Factor: {best_result['profit_factor']:.2f}")
            ]),
            dcc.Graph(figure=portfolio_fig)
        ])
        return results_layout

    except Exception as e:
        return html.P(f"An error occurred during optimization: {str(e)}", className="text-danger")

if __name__ == '__main__':
    app.run_server(debug=True)
