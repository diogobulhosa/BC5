from dis import code_info
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


import requests

url = 'https://web-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
list_crypto=[]
for start in range(1, 20000, 5000):

    params = {
        'start': start,
        'limit': 5000,
    }

    r = requests.get(url, params=params)
    data = r.json()
    for number, item in enumerate(data['data']):
        a = f"{item['symbol']}-USD"
        list_crypto.append(a)


def get_percentage_img(current_value, prev_value, height_size, prefix):
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        title = {"text": "1D Price Change"},
        value = current_value,
        delta = {'reference': prev_value, 'relative': True, 'valueformat':'.3%'},
        number={'valueformat':".5f",'prefix':prefix},
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



indicators_list = ['S&P 500', 'Dollar', 'Bollinger Bands','SMA30', 'CMA30', 'EMA30','Stochastic Oscillator', 'MACD', 'RSI', 'ADX', 'STDEV']
# testing? 
df_sp500_to_pass = yf.download('^GSPC', 
                      progress=False)
df_dollar_to_pass = yf.download('DX=F',  
                      progress=False)
    #####

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server


tab_portfolio =  html.Div([
    ## choose coin / target
    html.Div([
        html.Div([
            html.Div([
                html.H4('Choose crypto'),
                html.Br(),    
                html.H3(id='portfolio_value'),       
                ], className='box', style={'margin-top': '1%'}), # crypo choice over here
            html.Div([
                dcc.Dropdown(
                    id='main_coin_dropdown_portfolio',
                    value='BTC-USD',
                    multi=False,
                    options=list_crypto, 
                    style={'color': 'black', 'background-color':'#d3d3d3'}) ,
                html.Br(), 
                dbc.RadioItems(
                    id='buy_sell',
                    className='radio',
                    options=[dict(label='Buy', value=0), dict(label='Sell', value=1)],
                    inline=True
                    ), 
                html.Br(), 
                html.H4('Investment'),   
                dcc.Input(id="investment", type="text", placeholder="", debounce=True),
                html.Br(),             
                ], className='box', style={'margin-top': '1%','width': '60%', 'margin-left': '1%'}) # crypo choice over here
        ],className='info_box',style={'margin-left': '0%'}),
    ## predictions graph
    html.Div([
            html.Div([
                html.H4('Choose crypto'),
                html.Br(),
                ], className='box', style={'margin-top': '1%'}), # crypo choice over here
            html.Div([
                html.H4('Choose Target'), 
                html.Br(),                                  
                ], className='box', style={'margin-top': '1%','width': '60%', 'margin-left': '1%'}) # crypo choice over here
        ],className='info_box',style={'margin-left': '0%'})
    


    ],style={'margin-left': '0%'}), 
        
], className='main')


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
                options=list_crypto, 
                style={'color': 'black', 'background-color':'#d3d3d3'}) ,
            dcc.DatePickerRange(
                id = 'data_picker',
                min_date_allowed = date(2014, 9, 17),
                max_date_allowed = date.today(),
                start_date=date(2014, 9, 17),
                end_date=date.today(), 
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
                    ],className='box_crypto_info',style={'width': '25%'}),
            html.Div([
                        html.H4('Day Range', style={'font-weight':'bold'}),
                        html.H3(id='price_range')
                    ],className='box_crypto_info',style={'width': '25%'}),
            html.Div([
                        html.H4('Volume (%-w)', style={'font-weight':'bold'}),
                        dcc.Graph(id='volume_today2week_fig')
                    ],className='box_crypto_info',style={'width': '25%'}),
            html.Div([
                        html.H4('52-Week Range', style={'font-weight':'bold'}),
                        html.H3(id='price_range_weeks')
                    ],className='box_crypto_info',style={'width': '25%'}),             
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
                options=list_crypto, 
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
                    options=list_crypto, 
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
                    dbc.Tab(tab_portfolio, label="Portfolio", labelClassName ='labels', tabClassName = 'tabs', tab_style={'margin-left' : '0%'}),
                    dbc.Tab(tab_analysis, label="Analysis", labelClassName ='labels', tabClassName = 'tabs', tab_style={'margin-left' : '0%'}),
                    dbc.Tab(tab_predictions, label="Prediction", labelClassName ='labels', tabClassName = 'tabs', tab_style={'margin-left' : '0%'}),
                ])
        ],className='boxtabs', style={'margin-top': '0%', 'margin-left': '5%'}),
    ],
    fluid=True,
)

################################CALLBACK - Portfolio############################################
@app.callback(
    Output(component_id='portfolio_value', component_property='children'),
    [Input('main_coin_dropdown_portfolio', 'value'),
     Input('buy_sell', 'value'), 
     Input('investment', 'value')]
)

def callback_portfolio(coin_name, buy_sell, investment):
    print(coin_name, buy_sell, investment)
    df_portfolio = pd.read_csv (r'./portfolio_test.csv')
    df_time = pd.read_csv (r'./client_valuev2_test.csv')
    if buy_sell == 0: 
        signal = 'buy'
    else: signal = 'sell'

    portfolio_value = 10
    
    if (coin_name != None) and (investment != None):
       portfolio_value = 10

    
    return str(portfolio_value)



################################CALLBACK - Analysis############################################
@app.callback(
    [Output('data_picker', 'start_date'),
     Output('data_picker', 'min_date_allowed')],
    Input('main_coin_dropdown', 'value')
)

def callback_0(coin_name):
    df_coin = yf.download(coin_name,
                      progress=False,
    )
    crypto_first_day = df_coin.index.min()
    return crypto_first_day, crypto_first_day

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
    #print(coin_name,sec_coin_name, check_list, type(start_date), end_date)
    # create dataset
    df_coin = yf.download(coin_name,
                      progress=False,
    )
    df_coin_day = yf.download(coin_name,
                      start=date.today(), 
                      interval="1m",
                      progress=False,
    )

    df_coin = df_coin[df_coin.index >=start_date]

    # data conversion
    newstartdate = start_date[:10]
    newenddate=end_date[:10]
    a = datetime.strptime(str(newstartdate), '%Y-%m-%d')
    b = datetime.strptime(str(newenddate), '%Y-%m-%d')
    delta = b - a
    n_days = delta.days

    # testing? 
    #df_sp500 = yf.download('^GSPC', 
    #                  progress=False)
    #df_dollar = yf.download('DX=F',  
    #                  progress=False)
    #####

    df_sp500 = df_sp500_to_pass.copy()
    df_dollar = df_dollar_to_pass.copy()

    df_coin = fb.df_converter(df_coin, df_sp500,df_dollar)

    # first viz 
    today = pd.to_datetime(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(days=0))
    df_coin_day.index = df_coin_day.index.tz_localize(None)
    current_price = df_coin_day['Close'].iloc[-1]
    today_open_price_not_round = df_coin_day[df_coin_day.index >= today]['Open'][0]    
    curr_price_fig = get_percentage_img(current_price, today_open_price_not_round, 80, '$')


    #crypto first day in dataset
    crypto_first_day = str(df_coin.index.min())
    crypto_first_day = crypto_first_day[:10]
    crypto_first_day

    # open price today
    today_open_price = round(today_open_price_not_round,5)
    #print(today_open_price)

    #52-weeks 
    fiftytwo_weeks = pd.to_datetime(datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(weeks=52))
    high_fiftytwo_weeks=df_coin[df_coin.index >= fiftytwo_weeks]['High'].max()
    low_fiftytwo_weeks=df_coin[df_coin.index >= fiftytwo_weeks]['Low'].min()
    price_range_weeks = str(round(low_fiftytwo_weeks, 5))+' - '+str(round(high_fiftytwo_weeks, 5))

    # last volume 
    yesterday = pd.to_datetime(datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(days=1))
    weekly_yesterday = pd.to_datetime(datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(days=7))
    before_volume = df_coin[df_coin.index == weekly_yesterday]['Volume'][0]
    last_volume = df_coin[df_coin.index == yesterday]['Volume'][0]
    volume_today2week_fig = get_percentage_img(last_volume, before_volume, 50,'')

    # price range today
    today_high = df_coin_day[df_coin_day.index >= today]['High'].max()
    today_low = df_coin_day[df_coin_day.index >= today]['Low'].min()
    price_range = str(round(today_low, 5))+' - '+str(round(today_high, 5))
      

    # first viz 
    #print('aqui')
    fig = fb.candlestick(df_coin, days=n_days, indicators = check_list)

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

    df_sp500 = df_sp500_to_pass.copy()
    df_dollar = df_dollar_to_pass.copy()
    
    print('oi')
    df_coin = fb.df_converter(df_coin, df_sp500, df_dollar)
    model = 'XGB'
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
