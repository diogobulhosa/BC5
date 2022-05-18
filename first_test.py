import dash
from dash import dcc, dash_table
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import yfinance as yf
from datetime import date, timedelta

import finance_lib as fb


test_list = ['ADA-USD','ATOM-USD','AVAX-USD','AXS-USD','BTC-USD','ETH-USD','LINK-USD','LUNA1-USD','MATIC-USD','SOL-USD']


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

tab_analysis =  html.Div([
    ### choose circuit main information
    html.Div([
        html.Div([
            html.H4('Click on one of the red dots in the Map to select a circuit. (default: Circuit de Monaco)'),
            html.Br(), 
            dcc.Dropdown(
            id='main_coin_dropdown',
            value='BTC-USD',
            multi=False,
            options=test_list, 
            style={'color': 'black', 'background-color':'#d3d3d3'}) ,
            dcc.DatePickerRange(
                month_format='DoMMM Do, YY',
                end_date_placeholder_text='MMM Do, YY',
                start_date=date(2017, 6, 21)
            ),
            dcc.Graph(
                id='graph_price'
            )    
            
            ], className='box', style={'margin-top': '3%'})
    ],style={'margin-left': '0%'})
], className='main')

tab_predictions =  html.Div([
    
], className='main')


app.layout = dbc.Container([               
        html.Div([
            dbc.Tabs([
                    dbc.Tab(tab_analysis, label="Analysis", labelClassName ='labels', tabClassName = 'tabs'),
                    dbc.Tab(tab_predictions, label="Prediction", labelClassName ='labels', tabClassName = 'tabs', tab_style={'margin-left' : '0%'}),
                ])
        ],className='boxtabs', style={'margin-top': '3%', 'margin-left': '5%'}),
    ],
    fluid=True,
)

################################CALLBACK############################################

@app.callback(
    Output(component_id='graph_price', component_property='figure'),
    [Input('main_coin_dropdown', 'value')]
)

################################CALLBACKFUNCTIONCIRCUITS############################
# recebe os years, retorna o grafico+lista de circuitos available nessa altura
def callback_1(coin_name):
    # create dataset
    df_coin = yf.download(coin_name,
                      end=date.today() - timedelta(days=1), 
                      progress=False,
    )

    #df_coin = fb.df_converter(df_coin)

    print(df_coin.columns)

    # first viz 
    fig = px.line(df_coin, df_coin.index, y=df_coin['Close'])

    return fig
###################################################################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
