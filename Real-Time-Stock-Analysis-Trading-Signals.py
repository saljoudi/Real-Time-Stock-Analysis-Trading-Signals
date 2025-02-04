import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objs as go

from datetime import datetime
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings from TA library

def process_stock(df, stock_symbol, sma_short, sma_long, rsi_threshold, adl_short, adl_long, initial_investment=100000):
    # ------------------ 1) Moving Averages ------------------
    df['SMA_Short'] = df['Close'].rolling(window=int(sma_short)).mean()
    df['SMA_Long']  = df['Close'].rolling(window=int(sma_long)).mean()

    # ------------------ 2) RSI (Ensure 1D) ------------------
    rsi_series = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    # Convert to a true 1D Series
    rsi_series = rsi_series.squeeze()  # or rsi_series.iloc[:, 0] if it's DataFrame
    df['RSI'] = pd.Series(rsi_series.values, index=df.index)  # or just df['RSI'] = rsi_series if it's already Series

    # ------------------ 3) MACD (Ensure 1D) ------------------
    macd = ta.trend.MACD(df['Close'])
    df['MACD']        = macd.macd().squeeze()
    df['MACD_Signal'] = macd.macd_signal().squeeze()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # ------------------ 4) ADL (Ensure 1D) ------------------
    adl_series = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
    adl_series = adl_series.squeeze()
    df['ADL'] = pd.Series(adl_series.values, index=df.index)

    # ADL moving averages
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
            # Buy
            buy_price = row['Close']
            position = cash / buy_price
            amount_invested = cash
            cash = 0
            trade_start = index
            number_of_trades += 1

        elif row['Signal'] == -1 and position > 0:
            # Sell
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

# ---- The rest of your Dash app / callback code goes here, using process_stock() just as before ----



# -------------- Dash App ----------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server  # Expose the underlying WSGI server for deployment

app.layout = dbc.Container(fluid=True, children=[

    dbc.Row([
        dbc.Col(html.H1("Advanced Stock Analysis App", className="text-center text-primary mb-4"), width=12)
    ], className="mt-4"),

    dbc.Row([
        # Left column: input parameters
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Input Parameters"),
                dbc.CardBody([
                    dbc.Form([
                        html.Div([
                            dbc.Label("Stock Symbol:", html_for='stock-symbol'),
                            dbc.Input(
                                id='stock-symbol',
                                type='text',
                                value='AAPL',
                                placeholder='Enter stock symbol',
                                debounce=True
                            ),
                            dbc.FormText("Enter the stock ticker symbol (e.g., AAPL or 1303 for Saudi stocks)."),
                        ], className="mb-4"),

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
                                value='1d',  # default
                                placeholder='Select time period'
                            ),
                        ], className="mb-4"),

                        html.Div([
                            dbc.Label("SMA Short:", html_for='sma-short'),
                            dbc.Input(id='sma-short', type='number', value=7, min=1, debounce=True),
                        ], className="mb-4"),

                        html.Div([
                            dbc.Label("SMA Long:", html_for='sma-long'),
                            dbc.Input(id='sma-long', type='number', value=10, min=1, debounce=True),
                        ], className="mb-4"),

                        html.Div([
                            dbc.Label("RSI Threshold:", html_for='rsi-threshold'),
                            dbc.Input(id='rsi-threshold', type='number', value=40, min=0, max=100, debounce=True),
                        ], className="mb-4"),

                        html.Div([
                            dbc.Label("ADL Short:", html_for='adl-short'),
                            dbc.Input(id='adl-short', type='number', value=13, min=1, debounce=True),
                        ], className="mb-4"),

                        html.Div([
                            dbc.Label("ADL Long:", html_for='adl-long'),
                            dbc.Input(id='adl-long', type='number', value=30, min=1, debounce=True),
                        ], className="mb-4"),

                        dbc.Button('Analyze Stock', id='submit-button', color='primary', className='w-100'),
                    ])
                ])
            ], className="mb-5"),

            # Spinner placeholder for updates
            dbc.Spinner(html.Div(id="loading-output")),
        ], width=3),

        # Right column: tabs for outputs
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Price Chart", tab_id="tab-price-chart", children=[
                    dbc.CardBody([dcc.Graph(id='stock-graph', style={'height': '70vh'})])
                ]),
                dbc.Tab(label="MACD", tab_id="tab-macd", children=[
                    dbc.CardBody([dcc.Graph(id='macd-graph', style={'height': '70vh'})])
                ]),
                dbc.Tab(label="RSI", tab_id="tab-rsi", children=[
                    dbc.CardBody([dcc.Graph(id='rsi-graph', style={'height': '70vh'})])
                ]),
                dbc.Tab(label="Performance Metrics", tab_id="tab-metrics", children=[
                    dbc.CardBody([html.Div(id='performance-metrics', className="mt-3")])
                ]),
                dbc.Tab(label="Trade Details", tab_id="tab-trades", children=[
                    dbc.CardBody([html.Div(id='trade-details', className="mt-3")])
                ]),
            ], id='output-tabs', active_tab='tab-price-chart', className="mt-3"),
        ], width=9),
    ], className="mb-5"),

    # This Interval component will trigger the callback every 60 seconds
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # 60 seconds (in milliseconds)
        n_intervals=0
    )
])

# -------------- Callback ----------------
@app.callback(
    [
        Output('stock-graph', 'figure'),
        Output('macd-graph', 'figure'),
        Output('rsi-graph', 'figure'),
        Output('performance-metrics', 'children'),
        Output('trade-details', 'children'),
        Output('loading-output', 'children')
    ],
    [
        Input('interval-component', 'n_intervals'),  # triggers every 60s
        Input('submit-button', 'n_clicks')           # triggers on button click
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
def update_output(n_intervals, n_clicks, stock_symbol, time_period,
                  sma_short, sma_long, rsi_threshold, adl_short, adl_long):
    """
    This callback runs every 60 seconds or whenever the user clicks 'Analyze Stock'.
    It fetches the data for the chosen 'time_period' (1-5 days, 1-minute intervals),
    processes the signals, and returns:
      1) Price Chart figure
      2) MACD figure
      3) RSI figure
      4) Performance Metrics
      5) Trade Details
      6) Loading output placeholder
    """
    try:
        # If it's a numeric symbol, append .SR for Saudi market convention
        if stock_symbol.isdigit():
            stock_symbol += '.SR'

        # Fetch the data with the chosen time period (1m interval)
        df = yf.download(stock_symbol, period=time_period, interval='1m')
        df.index = pd.to_datetime(df.index)

        if df.empty:
            msg = f"No data found for {stock_symbol} over the last {time_period}."
            return dash.no_update, dash.no_update, dash.no_update, html.P(msg, className="text-danger"), "", ""

        # Process the stock data
        result = process_stock(df, stock_symbol, sma_short, sma_long, rsi_threshold, adl_short, adl_long)
        df = result['df']  # updated DataFrame with signals

        # ========== Price Chart ==========
        price_fig = go.Figure()

        price_fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines', name='Close Price',
            line=dict(color='blue', width=2)
        ))
        price_fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_Short'],
            mode='lines', name=f'SMA Short ({sma_short})',
            line=dict(color='orange', dash='dash', width=1)
        ))
        price_fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_Long'],
            mode='lines', name=f'SMA Long ({sma_long})',
            line=dict(color='green', dash='dot', width=1)
        ))

        # Buy/Sell signals
        buy_signals = df[df['Buy Signal']]
        sell_signals = df[df['Sell Signal']]

        price_fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['Close'],
            mode='markers', name='Buy Signal',
            marker=dict(symbol='triangle-up', color='green', size=10)
        ))
        price_fig.add_trace(go.Scatter(
            x=sell_signals.index, y=sell_signals['Close'],
            mode='markers', name='Sell Signal',
            marker=dict(symbol='triangle-down', color='red', size=10)
        ))

        price_fig.update_layout(
            title={'text': f"Price with Buy/Sell Signals - {stock_symbol}", 'x':0.5},
            xaxis_title='Date/Time',
            yaxis_title='Price',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            margin=dict(l=40, r=40, t=80, b=40),
        )

        # ========== MACD Chart ==========
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'],
            mode='lines', name='MACD', line=dict(color='blue')
        ))
        macd_fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_Signal'],
            mode='lines', name='MACD Signal', line=dict(color='orange')
        ))
        macd_fig.add_trace(go.Bar(
            x=df.index, y=df['MACD_Histogram'],
            name='MACD Histogram', marker_color='gray', opacity=0.5
        ))
        macd_fig.update_layout(
            title={'text': f"MACD - {stock_symbol}", 'x':0.5},
            xaxis_title='Date/Time',
            yaxis_title='MACD',
            template='plotly_white',
            margin=dict(l=40, r=40, t=80, b=40),
            showlegend=True
        )

        # ========== RSI Chart ==========
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            mode='lines', name='RSI', line=dict(color='purple')
        ))
        # Mark overbought / oversold regions if desired
        rsi_fig.add_shape(type='line', xref='paper', x0=0, x1=1, y0=70, y1=70,
                          line=dict(color='red', dash='dash'))
        rsi_fig.add_shape(type='line', xref='paper', x0=0, x1=1, y0=30, y1=30,
                          line=dict(color='green', dash='dash'))

        rsi_fig.update_layout(
            title={'text': f"RSI - {stock_symbol}", 'x':0.5},
            xaxis_title='Date/Time',
            yaxis_title='RSI',
            template='plotly_white',
            margin=dict(l=40, r=40, t=80, b=40),
            yaxis=dict(range=[0,100])
        )

        # ========== Performance Metrics ==========
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

        # ========== Trade Details ==========
        if trades:
            trades_df = pd.DataFrame(trades)

            # Format numeric columns
            trades_df['Buy Price'] = trades_df['Buy Price'].map('{:,.2f} SAR'.format)
            trades_df['Sell Price'] = trades_df['Sell Price'].map('{:,.2f} SAR'.format)
            trades_df['Amount Invested'] = trades_df['Amount Invested'].map('{:,.2f} SAR'.format)
            trades_df['Profit'] = trades_df['Profit'].map('{:,.2f} SAR'.format)
            trades_df['Profit Percentage'] = trades_df['Profit Percentage'].map('{:.2f}%'.format)
            # Round minutes to 2 decimals (optional)
            trades_df['Minutes Held'] = trades_df['Minutes Held'].map('{:.2f}'.format)

            # Convert times to strings
            trades_df['Buy Time'] = trades_df['Buy Time'].astype(str)
            trades_df['Sell Time'] = trades_df['Sell Time'].astype(str)

            trades_table = dbc.Table.from_dataframe(
                trades_df, striped=True, bordered=True, hover=True, responsive=True, className="mt-3"
            )
            trade_details = [
                html.H5("Trade Details"),
                trades_table
            ]
        else:
            trade_details = [html.P("No trades were made with the given parameters.")]

        return price_fig, macd_fig, rsi_fig, perf_metrics, trade_details, ""

    except Exception as e:
        return (
            dash.no_update, dash.no_update, dash.no_update,
            html.P(f"An error occurred: {str(e)}", className="text-danger mt-3"),
            "",
            ""
        )

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




