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

indicators_list = ['S&P 500', 'Dollar', 'Bollinger Bands','SMA30', 'CMA30', 'EMA30','Stochastic Oscillator', 'MACD', 'RSI', 'ADI', 'STDEV']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

tab_analysis =  html.Div([
    ### choose coin
    html.Div([
        html.Div([
            html.H4('Choose crypto'),
            html.Br(), 
            dcc.Dropdown(
                id='main_coin_dropdown',
                value='BTC-USD',
                multi=False,
                options=test_list, 
                style={'color': 'black', 'background-color':'#d3d3d3'}) ,
            dcc.DatePickerRange(
                id = 'data_picker',
                start_date=date(2017, 6, 21),
                display_format='DD-MM-YYYY',
                start_date_placeholder_text='DD-MM-YYYY',
                style={'color': 'black', 'background-color':'#d3d3d3'}
            )            
            ], className='box', style={'margin-top': '1%'})
    ],style={'margin-left': '0%'}),
    ### indicators and graph
    html.Div([
        #### indicators / choose crypto
        html.Div([
            html.H3('Secondary Coin'),
            dcc.Dropdown(
                id='second_coin_dropdown',
                multi=False,
                options=test_list, 
                style={'color': 'black', 'background-color':'#d3d3d3'}),
            html.Br(),
            html.H3('Financial Indicators'),
            dcc.Checklist(
                id='check_indicator',
                options=indicators_list,
                labelStyle={'display': 'block'},
            )
            ], className='box', style={'margin-top': '1%', 'width': '20%'}),
        #### graph
        html.Div([
            html.H4('Choose crypto'),
            html.Br(), 
            dcc.Graph(
                id='graph_price'
            )
            ], className='box', style={'margin-top': '1%', 'width': '80%', 'margin-left': '1%'}),
    ],className='circ_box',style={'margin-left': '0%'})

], className='main')

tab_predictions =  html.Div([
    ## choose coin / target
    html.Div([
        html.Div([
            html.Div([
                html.H4('Choose crypto'),
                html.Br(), 
                dcc.Dropdown(
                    id='main_coin_dropdown_pred',
                    value='BTC-USD',
                    multi=False,
                    options=test_list, 
                    style={'color': 'black', 'background-color':'#d3d3d3'}) ,
                dcc.DatePickerRange(
                    id = 'data_picker_pred',
                    start_date=date(2017, 6, 21),
                    display_format='DD-MM-YYYY',
                    start_date_placeholder_text='DD-MM-YYYY',
                    style={'color': 'black', 'background-color':'#d3d3d3'})            
                ], className='box', style={'margin-top': '1%'}), # crypo choice over here
            html.Div([
                html.H4('Choose Target'), 
                html.Br(),
                dbc.Row(
                    justify="center",
                    children=[
                        dbc.ButtonGroup(
                            style={"text-align": "center"},
                            children=[
                                dbc.Button(
                                    "Open",
                                    color="primary",
                                    n_clicks=0,
                                    id="white_color",
                                    active=False,
                                ),
                                dbc.Button(
                                    "Close",
                                    color="primary",
                                    n_clicks=0,
                                    id="black_color",
                                    active=False,
                        ),],),],),                    
                ], className='box', style={'margin-top': '1%','width': '60%', 'margin-left': '1%'}) # crypo choice over here
        ],className='circ_box',style={'margin-left': '0%'}),
    ## predictions graph
    html.Div([
            html.Div([
                html.H4('Choose crypto'),
                html.Br(),
                 #### graph
                dcc.Graph(
                    id='graph_pred'
                )          
                ], className='box', style={'margin-top': '1%'}), # crypo choice over here
            html.Div([
                html.H4('Choose Target'), 
                html.Br(),                                  
                ], className='box', style={'margin-top': '1%','width': '60%', 'margin-left': '1%'}) # crypo choice over here
        ],className='circ_box',style={'margin-left': '0%'})
    


    ],style={'margin-left': '0%'}), 
        
], className='main')


app.layout = dbc.Container([               
        html.Div([
            html.H1('Crypto Forecaster'),
            html.Hr(),
            dbc.Tabs([
                    dbc.Tab(tab_analysis, label="Analysis", labelClassName ='labels', tabClassName = 'tabs'),
                    dbc.Tab(tab_predictions, label="Prediction", labelClassName ='labels', tabClassName = 'tabs', tab_style={'margin-left' : '0%'}),
                ])
        ],className='boxtabs', style={'margin-top': '0%', 'margin-left': '5%'}),
    ],
    fluid=True,
)

################################CALLBACK - Analysis############################################

@app.callback(
    Output(component_id='graph_price', component_property='figure'),
    [Input('main_coin_dropdown', 'value'),
     Input('second_coin_dropdown', 'value'),
     Input('check_indicator', 'value'), 
     Input('data_picker', 'start_date'),
     Input('data_picker', 'end_date')]
)

################################CALLBACKFUNCTIONANLYSIS############################
def callback_1(coin_name, sec_coin_name, check_list, start_date, end_date):
    print(coin_name,sec_coin_name, check_list, type(start_date), end_date)
    # create dataset
    df_coin = yf.download(coin_name,
                      progress=False,
    )

    df_coin = fb.df_converter(df_coin)

    df_coin

    if sec_coin_name: 
       sec_df_coin = yf.download(sec_coin_name, 
                      progress=False,
    )


    #print(df_coin.columns)

    # first viz 
    fig = px.line(df_coin, df_coin.index, y=df_coin['Close'])
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        font_color="white",
        font_size= 15
    )

    return fig
###################################################################################################################
################################CALLBACK - Predictions############################################

@app.callback(
    Output(component_id='graph_pred', component_property='figure'),
    [Input('main_coin_dropdown_pred', 'value'),
     Input('data_picker_pred', 'start_date'),
     Input('data_picker_pred', 'end_date')]
)

################################CALLBACKFUNCTIONPREDICTIONS############################
def callback_2(coin_name, start_date_pred, end_date_pred):
    print('oi')
    print(coin_name, type(start_date_pred), end_date_pred)
    # create dataset
    df_coin = yf.download(coin_name,
                      progress=False,
    )

    df_coin = fb.df_converter(df_coin)
    #model = fb.choose_m    odel('RF')

    #fig2 = fb.predictions(df_coin,model, 5,100,'Close')
    
       # first viz 
    fig2 = px.line(df_coin, df_coin.index, y=df_coin['Close'])
    fig2.update_layout(
        template="plotly_dark",
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        font_color="white",
        font_size= 15
    )
    return fig2
###################################################################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
