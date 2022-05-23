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
from datetime import date, timedelta, datetime

import finance_lib as fb

def get_percentage_img(current_value, prev_value, height_size):
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        title = {"text": "1D Price Change"},
        value = current_value,
        delta = {'reference': prev_value, 'relative': True, 'valueformat':'.2%'},
        domain = {'x': [0, 1], 'y': [0, 1]}))
    fig.update_layout(
            template="plotly_dark",
            plot_bgcolor = 'rgba(0, 0, 0, 0)',
            paper_bgcolor = 'rgba(0, 0, 0, 0)',
            font_color="white",
            font_size= 500,
            margin={'t': 0,'l':0,'b':10,'r':0},
            height=height_size
        )

    return fig


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
            ),
            html.Div([
                dcc.Graph(id='latest_price')            
            ], className='price_box', style={'margin-top': '1%', 'margin-right':'80%'})     
            ], className='box', style={'margin-top': '1%'})
    ],style={'margin-left': '0%'}),

    ### coin info
    html.Div([
        html.Div([
            html.Div([
                        html.H4('Open', style={'font-weight':'bold'}),
                        html.H3(id='today_open_price')
                    ],className='box_crypto_info'),
            html.Div([
                        html.H4('Day Range', style={'font-weight':'bold'}),
                        html.H3(id='price_range')
                    ],className='box_crypto_info'),
            html.Div([
                        html.H4('Volume (%-w)', style={'font-weight':'bold'}),
                        dcc.Graph(id='volume_today2week_fig')
                    ],className='box_crypto_info'),
            html.Div([
                        html.H4('52-Week Range', style={'font-weight':'bold'}),
                        html.H3(id='price_range_weeks')
                    ],className='box_crypto_info'),             
            ], className='info_box', style={'margin-top': '1%'})
    ],style={'margin-left': '0%'}),
    ### indicators and graph
    html.Div([
        #### indicators / choose crypto
        html.Div([
            html.H3('Add Coin'),
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
    ],className='info_box',style={'margin-left': '0%'})

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
                ## daqui 
                dbc.RadioItems(
                    id='open_close',
                    className='radio',
                    options=[dict(label='Open', value=0), dict(label='Close', value=1)],
                    value=1, 
                    inline=True
                    )
                
                        ### aqui                   
                ], className='box', style={'margin-top': '1%','width': '60%', 'margin-left': '1%'}) # crypo choice over here
        ],className='info_box',style={'margin-left': '0%'}),
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
        ],className='info_box',style={'margin-left': '0%'})
    


    ],style={'margin-left': '0%'}), 
        
], className='main')


app.layout = dbc.Container([               
        html.Div([
            html.H1('INVESTMENT4ALL'),
            html.Hr(style={'borderWidth': "0.3vh", "width": "100%", "backgroundColor": "#B4E1FF","opacity":"1"}),
            html.H2('Crypto Forecaster'),
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
    [Output(component_id='graph_price', component_property='figure'),
     Output(component_id='latest_price', component_property='figure'),
     Output(component_id='today_open_price', component_property='children'),
     Output(component_id='price_range_weeks', component_property='children'),
     Output(component_id='volume_today2week_fig', component_property='figure'),
     Output(component_id='price_range', component_property='children')],
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
    df_coin_day = yf.download(coin_name,
                      start=date.today(), 
                      interval="1m",
                      progress=False,
    )

    #df_coin = fb.df_converter(df_coin)

    # first viz 
    today = pd.to_datetime(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(days=0))
    df_coin_day.index = df_coin_day.index.tz_localize(None)
    current_price = df_coin_day['Close'].iloc[-1]
    today_open_price_not_round = df_coin_day[df_coin_day.index == today]['Open'][0]
    curr_price_fig = get_percentage_img(current_price, today_open_price_not_round, 80)


    #crypto first day in dataset
    crypto_first_day = str(df_coin.index.min())
    crypto_first_day = crypto_first_day[:10]
    crypto_first_day

    # open price today
    today_open_price = str(round(df_coin_day[df_coin_day.index == today]['Open'][0],2))
    #print(today_open_price)

    #52-weeks 
    fiftytwo_weeks = pd.to_datetime(datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(weeks=52))
    high_fiftytwo_weeks=df_coin[df_coin.index >= fiftytwo_weeks]['High'].max()
    low_fiftytwo_weeks=df_coin[df_coin.index >= fiftytwo_weeks]['Low'].min()
    price_range_weeks = str(round(low_fiftytwo_weeks, 2))+' - '+str(round(high_fiftytwo_weeks, 2))

    # last volume 
    yesterday = pd.to_datetime(datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(days=1))
    weekly_yesterday = pd.to_datetime(datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(days=7))
    before_volume = df_coin[df_coin.index == weekly_yesterday]['Volume'][0]
    last_volume = df_coin[df_coin.index == yesterday]['Volume'][0]
    volume_today2week_fig = get_percentage_img(last_volume, before_volume, 50)

    # price range today
    today_high = df_coin_day[df_coin_day.index >= today]['High'].max()
    today_low = df_coin_day[df_coin_day.index >= today]['Low'].min()
    price_range = str(round(today_low, 2))+' - '+str(round(today_high, 2))
      

    # first viz 
    fig = px.line(df_coin, df_coin.index, y=df_coin['Close'])
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        font_color="white",
        font_size= 15
    )

    return fig,curr_price_fig,today_open_price,price_range_weeks,volume_today2week_fig,price_range
###################################################################################################################
################################CALLBACK - Predictions############################################

@app.callback(
    Output(component_id='graph_pred', component_property='figure'),
    [Input('main_coin_dropdown_pred', 'value'),
     Input('data_picker_pred', 'start_date'),
     Input('data_picker_pred', 'end_date'),
     Input('open_close', 'value'), 
     ]
)

################################CALLBACKFUNCTIONPREDICTIONS############################
def callback_2(coin_name, start_date_pred, end_date_pred, open_close):
    #print('oi')
    #print(coin_name, type(start_date_pred), end_date_pred)
    # create dataset
    #print(open_close)
    df_coin = yf.download(coin_name,
                      progress=False,
    )

    #df_coin = fb.df_converter(df_coin)
    model = fb.choose_model('XGB')

    fig2 = fb.predictions(df_coin,model, 5,100,'Close')

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
