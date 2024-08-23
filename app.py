#!/usr/bin/env python
# coding: utf-8



import sqlite3
import pandas as pd
import numpy as np
np.bool = np.bool_
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pykalman import KalmanFilter
from sklearn.linear_model import LinearRegression



import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_table
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


#Import Analytics Module File
import analytics_module as am




### Initialize Calculations Dataframes ###
price_data = am.fetch_tables_as_dataframe('stocks_data.db')
top_correlations = am.calc_correlations(price_data,'2023-01-01', '2023-12-31', display = 10)
cointegrations = am.calculate_cointegration_for_pairs(am.extract_asset_pairs(top_correlations), price_data,'2023-01-01', '2023-12-31')
vol_correlations = am.calc_vol_corr_pairs(am.extract_asset_pairs(top_correlations), price_data, '2023-01-01', '2023-12-31')
asset_options = [{"label": col, "value": col} for col in price_data.columns]





###################################### DASH GUI CODE ##########################################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])

# Define the header
header = dbc.Navbar(
    dbc.Container([
        # Navbar brand (App title)
        dbc.NavbarBrand("Strategy Research Tools", className="ms-2"),
        # Navbar items (e.g., links)
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Equities Pair Trading", href="/page-1", id = "page-1-link")),
                dbc.NavItem(dbc.NavLink("Index Regression", href="/page-2", id = "page-2-link")),
            ],
            className="ms-auto",  # Aligns nav items to the right
        ),
    ]),
    color="primary",    # Background color of the header
    dark=True,          # Use dark text (for light background)
    sticky="top",       # Makes the navbar stick to the top of the page
)

asset_1_list_dropdown = dbc.Col([
    html.Label("Select Stock 1:"),
    dcc.Dropdown(
        id="stock-1-dropdown",
        options=asset_options,
        placeholder="Select a Stock...",
        value=None,  # Default selected value (None means no selection)
        clearable=True,  # Allows the user to clear the selection
        style ={'width':"300px"}

    )
])

asset_2_list_dropdown = dbc.Col([
    html.Label("Select Stock 2:"),
    dcc.Dropdown(
        id="stock-2-dropdown",
        options=asset_options,
        placeholder="Select another Stock...",
        value=None,  # Default selected value (None means no selection)
        clearable=True,# Allows the user to clear the selection
        style ={'width':"300px"}
    )
])

eval_metric_dropdown = dbc.Col([
    html.Label("Evaluation_Metric:"),
    dcc.Dropdown(
        id="eval-metric-dropdown",
        options=[{"label": col, "value": col} for col in ['Total Return','Sharpe Ratio','Sortino Ratio','Calmar Ratio','Max Drawdown']],
        placeholder="Select Evaluation Metric...",
        value='Total Return',  # Default selected value (None means no selection)
        clearable=True,# Allows the user to clear the selection
        style ={'width':"300px"}
    )
])

eval_metric_dropdown_2 = dbc.Col([
    html.Label("Evaluation_Metric:"),
    dcc.Dropdown(
        id="eval-metric-dropdown-2",
        options=[{"label": col, "value": col} for col in ['Total Return','Sharpe Ratio','Sortino Ratio','Calmar Ratio','Max Drawdown']],
        placeholder="Select Evaluation Metric...",
        value='Total Return',  # Default selected value (None means no selection)
        clearable=True,# Allows the user to clear the selection
        style ={'width':"300px"}
    )
])

index_dropdown = dbc.Col([
    html.Label("Market Index"),
    dcc.Dropdown(
        id="index-dropdown",
        options=[{"label": "Nasdaq 100", "value": "NDX"}, {"label": "S&P 500", "value": "^SPX"},{"label": "Russell 2000", "value": "^RUT"} ],
        placeholder="Select Market Index...",
        value='Total Return',  # Default selected value (None means no selection)
        clearable=True,# Allows the user to clear the selection
        style ={'width':"300px"}
    )
])

index_dropdown_2 = dbc.Col([
    html.Label("Market Index"),
    dcc.Dropdown(
        id="index-dropdown-2",
        options=[{"label": "Nasdaq 100", "value": "NDX"}, {"label": "S&P 500", "value": "^SPX"},{"label": "Russell 2000", "value": "^RUT"} ],
        placeholder="Select Market Index...",
        value='Total Return',  # Default selected value (None means no selection)
        clearable=True,# Allows the user to clear the selection
        style ={'width':"300px"}
    )
])

asset_3_list_dropdown = dbc.Col([
    html.Label("Select Basket of Stocks:"),
    dcc.Dropdown(
        id="stock-3-dropdown",
        options=asset_options,
        placeholder="Select another Stock...",
        value=None,  # Default selected value (None means no selection)
        clearable=True,# Allows the user to clear the selection
        multi = True,
        style ={'width':"300px"}
    )
])


page_1_layout = dbc.Container([
    header,  # Add the header to the layout
    html.Br(),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Select Dates to Calc Correlations:"),
            html.Br(),
            dcc.DatePickerRange(
                id='date-range-picker',
                min_date_allowed=datetime.date(2019, 1, 1),
                max_date_allowed=datetime.datetime.today(),
                start_date=datetime.date(2024, 1, 1),
                end_date=datetime.datetime.today()
            )
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Number of Rows to Display:"),
            html.Br(),
            dcc.Input(
                id='num-rows-input',
                type='number',
                value=10,  # Default value
                min=1,
                style={'width': '10%'}
            ),
        ], width=12)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col(
            dash_table.DataTable(
                id='data-table',
                columns=[{"name": i, "id": i} for i in top_correlations.columns] + [{"name": i, "id": i} for i in cointegrations.columns if i == 'Cointegration P-Value']
                +[{"name": i, "id": i} for i in vol_correlations.columns if i == 'Vol Correlation'],
                data=[],
                page_size=10,  # Adjust the page size
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                {'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220, 220, 220)',}],
                sort_action='native',  # Enable sorting by clicking on column headers
                sort_mode='single',    # Allows sorting by one column at a time
            ),
            width=12
        )
    ]),
    html.Br(),
    html.Hr(),
    dbc.Row([
        html.Label("Enter Stock Tickers to Analyze:", style={'fontWeight': 'bold'}),
        html.Br(),        
        dbc.Col([
            asset_1_list_dropdown,
            html.Br(),  
            asset_2_list_dropdown,
            html.Br(),
            html.Label("Rolling Correlation (in days):"),
            html.Br(),  
            dcc.Slider(
                id='window-slider',
                min=5,
                max=100,
                step=1,
                value=30,
                marks={i: str(i) for i in range(5, 101, 10)},
                tooltip={"placement": "bottom", "always_visible": True}),
            html.Br(),  
            dbc.Button("Update", id='update-button', color="primary", className="mt-2")
        ], width = "auto"),
        dbc.Col(
            dcc.Graph(id='correlation-graph'),
            width=8)        
    ], className="mb-4"),
    html.Hr(),  
    html.Label("Z-Score Based Entry", style={'fontWeight': 'bold'}),
    dbc.Row(
            dbc.Col(
                dbc.Button("Toggle Collapse",
                id="collapse-button",
                className="ml-auto",
                n_clicks=1,),
                width={"size": 3, "offset": 9},  # Offsetting to align the button to the right
            )
        ),
    dbc.Collapse(
        dbc.Row([
            html.Br(),  
            html.Br(),
            dbc.Col([
                html.Div([
                    html.Label("Entry Threshold:"),
                    html.Br(),  
                    dcc.Slider(
                        id='z-entry-slider',
                        min=0,
                        max=4,
                        step=0.1,
                        value=2,
                        marks={i: str(i) for i in range(0, 5, 1)},
                        tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Exit Threshold:"),
                    html.Br(),  
                    dcc.Slider(
                        id='z-exit-slider',
                        min=0,
                        max=1,
                        step=0.1,
                        value=0,
                        marks={i: str(i) for i in np.arange(0, 2.5, 0.50)},
                        tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Rolling Z-Score Window:"),
                    dcc.Slider(
                        id='z-window-slider',
                        min=0,
                        max=100,
                        step=1,
                        value=30,
                        marks={i: str(i) for i in range(0, 101, 20)},
                        tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    dbc.Button("Update", id='update-button-z-score', color="primary", className="mt-2"),
                    html.Br(),
                    html.Br(),
                    dash_table.DataTable(
                        id='data-table-2',
                        columns=[{"name": i, "id": i} for i in ['Metric','Value']],
                        data=[],
                        page_size=10,  # Adjust the page size
                        style_table={'overflowX': 'auto', 'width':'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'},),
                ])
            ], width=3),
            dbc.Col([
                dcc.Graph(id='z-score-pnl-graph'),
            ], width={"size": 9, "offset": 0})
        ], justify="center"),
        id="collapse-section",
        is_open = True,
    ),      
    html.Hr(),
    html.Label("Z-Score Parameter Comparison", style={'fontWeight': 'bold'}),
    html.Br(),
    dbc.Row(
            dbc.Col(
                dbc.Button("Toggle Collapse",
                id="collapse-button-2",
                className="ml-auto",
                n_clicks=1,),
                width={"size": 3, "offset": 9},  # Offsetting to align the button to the right
            )
        ),
    dbc.Collapse(    
    dbc.Row([
            html.Br(),  
            html.Br(),
            dbc.Col([
                html.Div([
                    html.Label("Rolling Z-Score Window:"),
                    html.Br(),  
                    dcc.Slider(
                        id='z-window-slider-2',
                    min=0,
                    max=100,
                    step=1,
                    value=30,
                    marks={i: str(i) for i in range(0, 101, 20)},
                        tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    eval_metric_dropdown,
                    html.Br(),
                    dbc.Button("Update", id='update-button-heatmap', color="primary", className="mt-2"),
                    html.Br(),
                    html.Br(),
                    dash_table.DataTable(
                        id='data-table-z-comp',
                        columns=[{"name": i, "id": i} for i in ['Param','Value']],
                        data=[],
                        page_size=10,  # Adjust the page size
                        style_table={'overflowX': 'auto', 'width':'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'},),
                    html.Br(),
                ])
            ], width=3),
            dbc.Col([
                dcc.Graph(id='z-score-heatmap'),
            ], width={"size": 9, "offset": 0})
        ], justify="center"),
        id="collapse-section-2",
        is_open = True,
    ),        
    html.Hr(),
    html.Label("RSI Based Entry", style={'fontWeight': 'bold'}),
    dbc.Row(
            dbc.Col(
                dbc.Button("Toggle Collapse",
                id="collapse-button-3",
                className="ml-auto",
                n_clicks=1,),
                width={"size": 3, "offset": 9},  # Offsetting to align the button to the right
            )
        ),
    dbc.Collapse( 
    dbc.Row([
        html.Br(),  
        html.Br(),
        dbc.Col([
            html.Div([
                html.Label("RSI Short Entry Threshold:"),
                html.Br(),  
                dcc.Slider(
                    id='rsi-short-entry-slider',
                    min=50,
                    max=100,
                    step=1,
                    value=80,
                    marks={i: str(i) for i in range(50, 101, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                html.Label("RSI Long Entry Threshold:"),
                html.Br(),  
                dcc.Slider(
                    id='rsi-long-entry-slider',
                    min=0,
                    max=50,
                    step=1,
                    value=20,
                    marks={i: str(i) for i in range(0, 51, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                html.Label("RSI Exit Threshold:"),
                html.Br(),  
                dcc.Slider(
                    id='rsi-exit-slider',
                    min=40,
                    max=60,
                    step=1,
                    value=50,
                    marks={i: str(i) for i in range(40, 61, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                html.Label("RSI Lookback Window:"),
                dcc.Slider(
                    id='rsi-window-slider',
                    min=0,
                    max=100,
                    step=1,
                    value=14,
                    marks={i: str(i) for i in range(0, 101, 20)},
                    tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                dbc.Button("Update", id='update-button-rsi', color="primary", className="mt-2"),
                html.Br(),
                html.Br(),
                dash_table.DataTable(
                    id='data-table-3',
                    columns=[{"name": i, "id": i} for i in ['Metric','Value']],
                    data=[],
                    page_size=10,  # Adjust the page size
                    style_table={'overflowX': 'auto', 'width':'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'},),
            ])
        ], width=3),
        dbc.Col([
            dcc.Graph(id='rsi-pnl-graph'),
        ], width={"size": 9, "offset": 0})
    ], justify="center"),
        id="collapse-section-3",
        is_open = True,
    ), 
    html.Hr(),
    html.Label("RSI Parameter Comparison", style={'fontWeight': 'bold'}),
    dbc.Row(
            dbc.Col(
                dbc.Button("Toggle Collapse",
                id="collapse-button-4",
                className="ml-auto",
                n_clicks=1,),
                width={"size": 3, "offset": 9},  # Offsetting to align the button to the right
            )
        ),
    dbc.Collapse(     
    dbc.Row([
            html.Br(),  
            html.Br(),
            dbc.Col([
                html.Div([
                    html.Label("RSI Lookback Window:"),
                    html.Br(),  
                    dcc.Slider(
                        id='rsi-window-slider-2',
                    min=0,
                    max=100,
                    step=1,
                    value=14,
                    marks={i: str(i) for i in range(0, 101, 20)},
                        tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    eval_metric_dropdown_2,
                    html.Br(),
                    dbc.Button("Update", id='update-button-heatmap-2', color="primary", className="mt-2"),
                    html.Br(),
                    dash_table.DataTable(
                        id='data-table-rsi-comp',
                        columns=[{"name": i, "id": i} for i in ['Param','Value']],
                        data=[],
                        page_size=10,  # Adjust the page size
                        style_table={'overflowX': 'auto', 'width':'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'},),
                    html.Br(),
                    html.Br(),
                ])
            ], width=3),
            dbc.Col([
                dcc.Graph(id='rsi-heatmap'),
            ], width={"size": 9, "offset": 0})
        ], justify="center"),
        id="collapse-section-4",
        is_open = True,
    ), 
])


###### PAGE 2 LAYOUT #############

page_2_layout = dbc.Container([
    header,
    html.Hr(),
    dbc.Row([
        html.Label("Enter Index to Analyze:", style={'fontWeight': 'bold'}),
        html.Br(),        
        dbc.Col([
            index_dropdown,
        ])
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            asset_3_list_dropdown,
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button("Update Plot", id="update-button", color="primary", className="mt-2")
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='regression-plot'),
            html.Br(),
            html.Div(id='regression-table')
        ])
    ]),
    html.Hr(),
    dbc.Row([
        html.Label("Index Explanation", style={'fontWeight': 'bold'}),
        html.Br(),        
        dbc.Col([
            index_dropdown_2,
            html.Br(),
            html.Div(id='pca-table')
        ])
    ]),
    
    
])



# Define the layout for the entire app
app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])


# Define the callback to update the page content based on the URL
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def display_page(pathname):
    if pathname == "/page-1":
        return page_1_layout
    elif pathname == "/page-2":
        return page_2_layout
    else:
        return page_1_layout  # Default to Page 1 if no match


# Callback to update the data table when the date range is changed

@app.callback(
    Output('pca-table','children'),
    Input('index-dropdown-2', 'value'),
    )

def update_pca_table(index_value):
    if not index_value:
        return {}    
    
    index_returns = price_data[index_value]
    stock_returns = price_data.loc[:, ~price_data.columns.isin(['NDX','^SPX','^RUT'])]
    
    output_df = am.pca_analysis(index_returns,stock_returns,variance_threshold = 0.90)    
    
    table_html = dbc.Table.from_dataframe(output_df, striped=True, bordered=True, hover=True)

    return table_html   


@app.callback(
    Output('regression-plot', 'figure'),
    Output('regression-table','children'),
    Input('update-button', 'n_clicks'),
    State('stock-3-dropdown', 'value'),
    State('index-dropdown','value')
)
def update_regression_plot(n_clicks, selected_tickers, index):
    if not selected_tickers:
        return {}

    # Fetch data from Yahoo Finance
    ticker_data = price_data[selected_tickers + [index]]
    data = ticker_data.pct_change().dropna()

    # Prepare the data for regression
    X = data[selected_tickers].values
    y = data[index].values

    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X, missing='drop').fit()

    # Get predictions
    predictions = model.predict(X)

    # Create the scatter plot
    fig = px.scatter(x=predictions, y=y, labels={'x': 'Predicted Index Returns', 'y': 'Actual Index Returns'},
                         title='Index Performance vs. Predicted Performance')
    fig.add_traces(px.line(x=predictions, y=predictions, labels={'x': '', 'y': ''}).data)  # Add the regression line

    # Create regression output table
    summary = model.summary2().tables[1]
    summary_df = pd.DataFrame(summary)
    summary_df.reset_index(inplace=True)
    summary_df.columns = ['Variable', 'Coefficient', 'Std Err', 't Value', 'P Value','Lower Est.','Upper Est.']
    summary_df['Variable'] = ['Constant']+ selected_tickers 
    summary_df = summary_df.round(6)
    table_html = dbc.Table.from_dataframe(summary_df, striped=True, bordered=True, hover=True)
    
    return fig, table_html


@app.callback(
    Output('data-table', 'data'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('num-rows-input', 'value')]
)
def update_table(start_date, end_date,num_rows):
    # Ensure num_rows is an integer and not less than 1
    if num_rows is None or int(num_rows) < 1:
        num_rows = 1    
    
    # Regenerate the DataFrame (this could be based on some logic or user input)
    correlations = am.calc_correlations(price_data,start_date,end_date,display = int(num_rows))     
    coints =  am.calculate_cointegration_for_pairs(am.extract_asset_pairs(correlations), price_data,start_date, end_date)
    vol_corr = am.calc_vol_corr_pairs(am.extract_asset_pairs(correlations), price_data,start_date, end_date)
    updated_df = correlations.merge(coints, how = 'left',on = ['Stock 1','Stock 2'])
    updated_df = updated_df.merge(vol_corr, how = 'left', on = ['Stock 1','Stock 2'])
    return updated_df.to_dict('records')

@app.callback(
    Output('correlation-graph', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('stock-1-dropdown', 'value'),
     State('stock-2-dropdown', 'value'),
     State('window-slider', 'value')]
)

def update_graph(n_clicks, asset1, asset2, window):
    # Prevent update if no button click
    if n_clicks is None:
        return dash.no_update
    
    # Calculate the rolling correlation
    rolling_corr = am.calculate_rolling_correlation(price_data, asset1, asset2, window)
    
    # Create the figure
    fig = {
        'data': [
            {
                'x': rolling_corr.index,
                'y': rolling_corr,
                'type': 'line',
                'name': f'Rolling Correlation ({asset1} vs {asset2})'
            }
        ],
        'layout': {
            'title': {
                'text': f'Rolling {window}-Day Correlation between {asset1} and {asset2}',
                'font': {
                    'size': 16  # Change this value to adjust the font size
                },
                'x': 0.5,  # Center the title
            },
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Correlation'},
            'template': 'plotly_dark'
        }
    }    
    return fig

@app.callback(
    Output('data-table-2','data'),
    [Input('update-button-z-score', 'n_clicks')],
    [State('stock-1-dropdown', 'value'),
     State('stock-2-dropdown', 'value'),
     State('z-entry-slider', 'value'),
     State('z-exit-slider', 'value'),
     State('z-window-slider', 'value')
    ]
)

def update_table(n_clicks, asset1, asset2, z_entry, z_exit, z_window):
    if n_clicks is None:
        return dash.no_update
    
    # Calculate the PnL
    strat =  am.simulate_pairs_trading_strategy(price_data, asset1, asset2, z_entry, z_exit, z_window)

    pnl = strat['PnL%']
    dates = strat['Date']

    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25  
    metrics = am.calculate_metrics(years,pnl)

    return metrics.to_dict('records')

@app.callback(
    Output('z-score-pnl-graph', 'figure'),
    [Input('update-button-z-score', 'n_clicks')],
    [State('stock-1-dropdown', 'value'),
     State('stock-2-dropdown', 'value'),
     State('z-entry-slider', 'value'),
     State('z-exit-slider', 'value'),
     State('z-window-slider', 'value')
    ]
)


def update_graph(n_clicks, asset1, asset2, z_entry, z_exit, z_window):
    # Prevent update if no button click
    if n_clicks is None:
        return dash.no_update
    
    pnl = am.simulate_pairs_trading_strategy(price_data, asset1, asset2, z_entry, z_exit, z_window)['PnL%']

    # Create the figure
    fig = {
        'data': [
            {
                'x': pnl.index,
                'y': np.cumsum(pnl),
                'type': 'line',
                'name': f'Cumulative PnL ({asset1} vs {asset2})'
            }
        ],
        'layout': {
            'title': {
                'text': f'Cumulative PnL: {asset1}/{asset2} using {z_entry} Z-Score Entry',
                'font': {
                    'size': 16  # Change this value to adjust the font size
                },
                'x': 0.5,  # Center the title
            },
            'xaxis': {'title': 'Number of Trades'},
            'yaxis': {'title': 'Cumulative PnL (%)'},
            'template': 'plotly_dark'
        }
    }    
    return fig


@app.callback(
    Output('z-score-heatmap', 'figure'),
    Output('data-table-z-comp','data'),
    [Input('update-button-heatmap', 'n_clicks')],
    [State('stock-1-dropdown', 'value'),
     State('stock-2-dropdown', 'value'),
     State('z-window-slider-2', 'value'),
     State('eval-metric-dropdown', 'value'),
    ] 
)
def update_heatmap_and_table(n_clicks, asset1, asset2, rolling_window_2, eval_metric):
    if n_clicks is None:
        return dash.no_update
    
    df = am.parameter_performance(price_data,asset1,asset2,rolling_window_2, eval_metric) 
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Use helper function to calculate optimal set of parameters
    params_df = am.optimal_conditions_table(df,rolling_window_2,eval_metric)
 
    fig = px.imshow(df,color_continuous_scale = "rdylgn",
    labels=dict(x="Entry Z-Score", y="Exit Z-Score", color=eval_metric))
    fig.update_layout(title="Strategy Performance Heatmap", title_x=0.5)
    fig.show()
    
    return fig, params_df.to_dict('records')

######## UPDATE RSI STRATEGY COMPONENTS ###################

@app.callback(
    Output('data-table-3','data'),
    [Input('update-button-rsi', 'n_clicks')],
    [State('stock-1-dropdown', 'value'),
     State('stock-2-dropdown', 'value'),
     State('rsi-long-entry-slider', 'value'),
     State('rsi-short-entry-slider', 'value'),
     State('rsi-exit-slider', 'value'),
     State('rsi-window-slider', 'value')
    ]
)

def update_table(n_clicks, asset1, asset2, rsi_long, rsi_short, rsi_exit, rsi_window):
    if n_clicks is None:
        return dash.no_update
    
    # Calculate the PnL
    strat =  am.simulate_rsi_trading_strategy(price_data, asset1, asset2, rsi_short,rsi_long,rsi_exit,rsi_window)

    pnl = strat['PnL%']
    dates = strat['Entry Date']

    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25  

    metrics = am.calculate_metrics(years,pnl)


    return metrics.to_dict('records')

@app.callback(
    Output('rsi-pnl-graph', 'figure'),
    [Input('update-button-rsi', 'n_clicks')],
    [State('stock-1-dropdown', 'value'),
     State('stock-2-dropdown', 'value'),
     State('rsi-long-entry-slider', 'value'),
     State('rsi-short-entry-slider', 'value'),
     State('rsi-exit-slider', 'value'),
     State('rsi-window-slider', 'value')
    ]
)

def update_graph(n_clicks, asset1, asset2, rsi_long, rsi_short, rsi_exit, rsi_window):
    # Prevent update if no button click
    if n_clicks is None:
        return dash.no_update
    
    pnl = am.simulate_rsi_trading_strategy(price_data, asset1, asset2, rsi_short,rsi_long,rsi_exit,rsi_window)['PnL%']

    # Create the figure
    fig = {
        'data': [
            {
                'x': pnl.index,
                'y': np.cumsum(pnl),
                'type': 'line',
                'name': f'Cumulative PnL ({asset1} vs {asset2})'
            }
        ],
        'layout': {
            'title': {
                'text': f'Cumulative PnL: {asset1}/{asset2} using {rsi_short} RSI Short Entry & {rsi_long} RSI Long Entry',
                'font': {
                    'size': 16  # Change this value to adjust the font size
                },
                'x': 0.5,  # Center the title
            },
            'xaxis': {'title': 'Number of Trades'},
            'yaxis': {'title': 'Cumulative PnL (%)'},
            'template': 'plotly_dark'
        }
    }    
    return fig

@app.callback(
    Output('rsi-heatmap', 'figure'),
    Output('data-table-rsi-comp','data'),
    [Input('update-button-heatmap-2', 'n_clicks')],
    [State('stock-1-dropdown', 'value'),
     State('stock-2-dropdown', 'value'),
     State('rsi-window-slider-2', 'value'),
     State('eval-metric-dropdown-2', 'value'),
    ] 
)
def update_heatmap_and_table(n_clicks, asset1, asset2, rolling_window_2, eval_metric):
    if n_clicks is None:
        return dash.no_update
    
    df = am.rsi_parameter_performance(price_data,asset1,asset2,rolling_window_2, eval_metric) 
    df = df.apply(pd.to_numeric, errors='coerce')
    
    params_df = am.optimal_conditions_table_rsi(df,rolling_window_2,eval_metric)

 
    fig = px.imshow(df,color_continuous_scale = "rdylgn",
    labels=dict(x="RSI Short Level", y="RSI Long Level", color=eval_metric))
    fig.update_layout(title="Strategy Performance Heatmap", title_x=0.5)
    fig.show()
    
    return fig,params_df.to_dict('records')

@app.callback(
    Output("collapse-section", "is_open"),
    Input("collapse-button", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_collapse(n_clicks):
    if n_clicks % 2 == 1:
        return True
    else:
        return False
    
@app.callback(
    Output("collapse-section-2", "is_open"),
    Input("collapse-button-2", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_collapse(n_clicks):
    if n_clicks % 2 == 1:
        return True
    else:
        return False

@app.callback(
    Output("collapse-section-3", "is_open"),
    Input("collapse-button-3", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_collapse(n_clicks):
    if n_clicks % 2 == 1:
        return True
    else:
        return False
    
@app.callback(
    Output("collapse-section-4", "is_open"),
    Input("collapse-button-4", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_collapse(n_clicks):
    if n_clicks % 2 == 1:
        return True
    else:
        return False

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)