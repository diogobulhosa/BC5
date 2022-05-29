from dis import code_info
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
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

########################### DASH UTILS #################################
indicators_list = ['S&P 500', 'Dollar', 'Bollinger Bands','SMA30', 'CMA30', 'EMA30','Stochastic Oscillator', 'MACD', 'RSI', 'ADX', 'STDEV']
# testing? 
df_sp500_to_pass = yf.download('^GSPC', 
                      progress=False)
df_dollar_to_pass = yf.download('DX=F',  
                      progress=False)
#####
df_summary_to_pass = pd.DataFrame(columns=['Coin', 'Percentage','Value','Spent'])
df_transactions_to_pass = pd.DataFrame(columns=['Coin', 'Date', 'Percentage', 'Value'])
made_purchase = 0
#########################################################################
########################### DASH FUNCTIONS ##############################
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

def atualization(df_summary):
    value_list=[]
    for idx,coins in enumerate(df_summary['Coin'].unique()):
        df_coin = yf.download(coins, progress=False)
        value_to_add = df_coin.tail(1)['Close'][0] * df_summary[df_summary['Coin'] == coins]['Percentage'][idx]
        value_list.append(round(value_to_add,2))
    
    df_summary.drop('Value', axis=1, inplace=True)
    df_summary['Value'] = value_list
    return df_summary
##########################################################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

tab_portfolio =  html.Div([
    ## choose coin / target
    html.Div([
        html.Div([
            html.Div([
                html.Button('Start New Portfolio', id='start_portfolio'),
                html.H4('Click here to reset portfolio'),
                dcc.Store(id='store_summary', data =[], storage_type = 'memory'),
                dcc.Store(id='store_transactions', data =[], storage_type = 'memory'),  
                dcc.Store(id='portfolio_state', data =[], storage_type = 'memory'), 
                html.Br(),    
                html.H3(id='portfolio_value'),
                html.H2(id='Your Portfolio'),    
                dcc.Graph(id='portfolio_pie')
                ], className='box', style={'margin-top': '1%'}), # crypo choice over here
            html.Div([
                dcc.Dropdown(
                    id='main_coin_dropdown_portfolio',
                    value='BTC-USD',
                    multi=False,
                    options=list_crypto, 
                    style={'color': 'black', 'background-color':'#d3d3d3'}) ,
                dcc.DatePickerSingle(
                    id='date_bought_portfolio',
                    max_date_allowed=date.today(),
                    display_format='DD-MM-YYYY',
                    date=date(2020, 8, 25)), # change this
                html.Br(), 
                html.Br(), 
                html.Br(), 
                html.H4('Investment'),   
                dcc.Input(id="investment", type="text", placeholder="", debounce=True),
                html.Br(),         
                html.Button('Make Purchase', id='make_purchase'), 
                ## daqui
                html.Div([
                    html.Div([
                        html.Div([
                                    html.H4('AVG Investment', style={'font-weight':'bold'}),
                                    html.H3(id='avg_invest')
                                ],className='box_crypto_info',style={'width': '25%'}),
                        html.Div([
                                    html.H4('Biggest Investment', style={'font-weight':'bold'}),
                                    html.H3(id='big_invest')
                                ],className='box_crypto_info',style={'width': '25%'}),
                        html.Div([
                                    html.H4('Total Investments', style={'font-weight':'bold'}),
                                    html.H3(id='total_invest')
                                ],className='box_crypto_info',style={'width': '25%'}),
                        html.Div([
                                    html.H4('Portfolio Value', style={'font-weight':'bold'}),
                                    dcc.Graph(id='port_total_value')
                                ],className='box_crypto_info',style={'width': '25%'}),            
                        ], className='info_box', style={'margin-top': '5%'})
                ],style={'margin-left': '0%'}),
                ## aqui 
                dcc.Graph(id='porto_histogram', style={'margin-top':'5%'})
               
                ], className='box', style={'margin-top': '1%','width': '60%', 'margin-left': '1%'}) # crypo choice over here
        ],className='info_box',style={'margin-left': '0%'}),
    

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
                html.H4('Choose crypto, dates to use for prediction and number of days to predict.'),
                dcc.Dropdown(
                    id='main_coin_dropdown_pred',
                    value='BTC-USD',
                    multi=False,
                    options=list_crypto, 
                    style={'color': 'black', 'background-color':'#d3d3d3'}) ,
                dcc.DatePickerRange(
                    id = 'data_picker_pred',                    
                    max_date_allowed = date.today(),
                    end_date=date.today(),
                    display_format='DD-MM-YYYY',
                    start_date_placeholder_text='Start Date',
                    style={'color': 'black', 'background-color':'#d3d3d3'}),
                html.Br(), 
                html.Br(),                
                dcc.Slider(5, 10, 1,
                    value=5,
                    id='days_to_pred'),
                ##            
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
                html.H4('Predictions Graph'),
                html.Br(),
                 #### graph
                dcc.Graph(
                    id='graph_pred'
                )          
                ], className='box', style={'margin-top': '1%'}), # crypo choice over here
            html.Div([
                html.H4('Metrics'), 
                html.Div([
                    html.H4('RMSE', style={'font-weight':'bold'}),
                    html.H3(id='rmse')
                        ],className='box_crypto_info',style={'width': '100%', 'justify-content': 'center'}),
                html.Div([
                    html.H4('MAE', style={'font-weight':'bold'}),
                    html.H3(id='mae')
                        ],className='box_crypto_info',style={'width': '100%', 'justify-content': 'center'}),
                html.Div([
                        html.H4('R Squared', style={'font-weight':'bold'}),
                        html.H3(id='rsquared')
                            ],className='box_crypto_info',style={'width': '100%', 'justify-content': 'center'}),
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
                ]),
                #ate aqui
            html.Div([
                html.Div([
                    html.Br(),
                    html.P(['GroupW', html.Br(),'Diogo Bulhosa (20210601), Francisco Costa (20211022), Mafalda Figueiredo (20210591), Rodrigo Pimenta (20210599)'], style={'font-size':'12px'}),
                ], style={'width':'100%'}), 
            ], className = 'footer', style={'display':'flex'}),
        ],className='boxtabs', style={'margin-top': '0%', 'margin-left': '5%'}),
    ],

    fluid=True,
)

################################CALLBACK - Portfolio############################################
@app.callback(
    [Output(component_id='portfolio_pie', component_property='figure'),
     Output(component_id='store_transactions', component_property='data'),
     Output(component_id='store_summary', component_property='data'), 
     Output(component_id='portfolio_state', component_property='data'),
     Output(component_id='total_invest', component_property='children'),
     Output(component_id='port_total_value', component_property='figure'),
     Output(component_id='avg_invest', component_property='children'),
     Output(component_id='big_invest', component_property='children'),
     Output(component_id='porto_histogram', component_property='figure'),],
    [Input('start_portfolio', 'n_clicks'),
     Input('store_summary', 'data'),
     Input('store_transactions', 'data'),
     Input('main_coin_dropdown_portfolio', 'value'),
     Input('date_bought_portfolio', 'date'),
     Input('investment', 'value'),
     Input('make_purchase', 'n_clicks'), 
     Input('portfolio_state', 'data')]
)

def callback_portfolio_create(start_portfolio, dict_summary, dict_transactions, coin_name, portfolio_date, investment, n_purchases, portfolio_state):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'start_portfolio' in changed_id:
        print('Reiniciar Portfolio') 
        df_summary = df_summary_to_pass.copy()
        df_transactions = df_transactions_to_pass.copy()

        fig = go.Figure(px.pie(labels=df_summary['Coin'],
                                values=df_summary['Value'],
                                names = df_summary['Coin'],
                                hole=.4,color_discrete_sequence=px.colors.qualitative.T10))
        fig.update_layout(width=500, height=500,paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)', font_color = 'white')
        n_purchases=0
        avg_invest=0
        big_invest=0
        port_total_value = 0 
        invest_total_value = 0
        portfolio_flunctuation_fig = get_percentage_img(port_total_value, invest_total_value, 50, prefix = '$')

        color_discrete_map = {'Spent': 'rgb(30,144,255)', 'Value': 'rgb(0,0,255)'}
        fig2 = px.histogram(df_summary, x="Coin", y=['Spent',"Value"],
                    barmode='group',color_discrete_map=color_discrete_map,
                    height=400)

        fig2.update_layout(
                template="plotly_dark",
                plot_bgcolor = 'rgba(0, 0, 0, 0)',
                paper_bgcolor = 'rgba(0, 0, 0, 0)',
                font_color="white",
                font_size= 10,
                margin={'t': 0,'l':0,'b':10,'r':0})
        
        return fig, df_transactions.to_dict(orient='records'),df_summary.to_dict(orient='records'), portfolio_state, invest_total_value, portfolio_flunctuation_fig, avg_invest, big_invest,fig2   
    else: 
        # se tentou comprar
        if 'make_purchase' in changed_id:
            # se conseguir fazer investimento faz
            if (coin_name!=None) and (portfolio_date!=None) and (investment!=None) and (n_purchases!=None):
                # # se dicionários não vazios cria transações ou dá update
                if dict_transactions:      
                        print('Investimento feito') 
                        df_summary = pd.DataFrame.from_dict(dict_summary)
                        df_transactions = pd.DataFrame.from_dict(dict_transactions)
                        
                        newportdate = portfolio_date[:10]
                        portdate = pd.to_datetime(newportdate, format='%Y-%m-%d')
                        df_sp500 = df_sp500_to_pass.copy()
                        df_dollar = df_dollar_to_pass.copy()

                        df_coin = yf.download(coin_name, progress=False)
                        df_coin = fb.df_converter(df_coin,df_sp500,df_dollar)


                        df_transactions, df_summary = fb.portofolio(df_transactions,float(investment),portdate,df_coin,coin_name,df_summary)
                        df_summary = atualization(df_summary)

                        data = [go.Pie(labels=df_summary['Coin'],
                                values=df_summary['Value'],
                                hole=.4)]

                        fig = go.Figure(px.pie(labels=df_summary['Coin'],
                                values=df_summary['Value'],
                                names = df_summary['Coin'],
                                hole=.4,color_discrete_sequence=px.colors.qualitative.T10))
                        fig.update_layout(width=800, height=600,paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)', font_color = 'white')
                        
                        avg_invest = round(df_transactions['Value'].mean(),2)
                        big_invest = df_transactions['Value'].max()
                        port_total_value = df_summary['Value'].sum()
                        invest_total_value = df_transactions['Value'].sum()
                        portfolio_flunctuation_fig = get_percentage_img(port_total_value, invest_total_value, 50, prefix = '$')
                        color_discrete_map = {'Spent': 'rgb(30,144,255)', 'Value': 'rgb(0,0,255)'}
                        fig2 = px.histogram(df_summary, x="Coin", y=['Spent',"Value"],
                                    barmode='group',color_discrete_map=color_discrete_map,
                                    height=400)

                        fig2.update_layout(
                                template="plotly_dark",
                                plot_bgcolor = 'rgba(0, 0, 0, 0)',
                                paper_bgcolor = 'rgba(0, 0, 0, 0)',
                                font_color="white",
                                font_size= 10,
                                margin={'t': 0,'l':0,'b':10,'r':0})
                        return fig, df_transactions.to_dict(orient='records'),df_summary.to_dict(orient='records'), portfolio_state, invest_total_value, portfolio_flunctuation_fig, avg_invest, big_invest,fig2
                else:      
                        print('Investimento feito -> criar dicionário por ser o primeiro') 
                        df_summary = df_summary_to_pass.copy()
                        df_transactions = df_transactions_to_pass.copy()
                        newportdate = portfolio_date[:10]
                        portdate = pd.to_datetime(newportdate, format='%Y-%m-%d')
                        df_sp500 = df_sp500_to_pass.copy()
                        df_dollar = df_dollar_to_pass.copy()

                        df_coin = yf.download(coin_name, progress=False)
                        df_coin = fb.df_converter(df_coin,df_sp500,df_dollar)


                        df_transactions, df_summary = fb.portofolio(df_transactions,float(investment),portdate,df_coin,coin_name,df_summary)
                        df_summary = atualization(df_summary)

                        data = [go.Pie(labels=df_summary['Coin'],
                                values=df_summary['Value'],
                                hole=.4)]

                        fig = go.Figure(px.pie(labels=df_summary['Coin'],
                                values=df_summary['Value'],
                                names = df_summary['Coin'],
                                hole=.4,color_discrete_sequence=px.colors.qualitative.T10))
                        fig.update_layout(width=800, height=600,paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)', font_color = 'white')
                        avg_invest = round(df_transactions['Value'].mean(),2)
                        big_invest = df_transactions['Value'].max()
                        port_total_value = df_summary['Value'].sum()
                        invest_total_value = df_transactions['Value'].sum()
                        portfolio_flunctuation_fig = get_percentage_img(port_total_value, invest_total_value, 50, prefix = '$')
                        color_discrete_map = {'Spent': 'rgb(30,144,255)', 'Value': 'rgb(0,0,255)'}
                        fig2 = px.histogram(df_summary, x="Coin", y=['Spent',"Value"],
                                    barmode='group',color_discrete_map=color_discrete_map,
                                    height=400)

                        fig2.update_layout(
                                template="plotly_dark",
                                plot_bgcolor = 'rgba(0, 0, 0, 0)',
                                paper_bgcolor = 'rgba(0, 0, 0, 0)',
                                font_color="white",
                                font_size= 10,
                                margin={'t': 0,'l':0,'b':10,'r':0})
                        return fig, df_transactions.to_dict(orient='records'),df_summary.to_dict(orient='records'), portfolio_state, invest_total_value, portfolio_flunctuation_fig, avg_invest, big_invest, fig2
            else: 
                print('Investimento por fazer: waiting') 
                return dash.no_update # atualizar para só faz update
        else:
            print('Não tentou comprar')
            return dash.no_update
    
########################################################################################
################################CALLBACK - Analysis#####################################
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
      
    if sec_coin_name != None:
        sec_coin_name = yf.download(sec_coin_name,
                    progress=False,
        )

    # first viz 
    #print('aqui')

    fig = fb.candlestick(df_coin,comparison=sec_coin_name, days=n_days, indicators = check_list)



    return fig,curr_price_fig,today_open_price,price_range_weeks,volume_today2week_fig,price_range
###################################################################################################################
################################CALLBACK - Predictions############################################

@app.callback(
    [Output(component_id='graph_pred', component_property='figure'),
     Output(component_id='rmse', component_property='children'),
     Output(component_id='mae', component_property='children'),
     Output(component_id='rsquared', component_property='children')],
    [Input('main_coin_dropdown_pred', 'value'),
     Input('data_picker_pred', 'start_date'),
     Input('data_picker_pred', 'end_date'),
     Input('open_close', 'value'), 
     Input('days_to_pred', 'value'), 
     ]
)

################################CALLBACKFUNCTIONPREDICTIONS############################
def callback_2(coin_name, start_date_pred, end_date_pred, open_close, days_to_pred):
    df_coin = yf.download(coin_name,
                      progress=False,
    )

    print(coin_name,start_date_pred,end_date_pred,open_close,days_to_pred)

    df_sp500 = df_sp500_to_pass.copy()
    df_dollar = df_dollar_to_pass.copy()


    if (coin_name != None) and (start_date_pred != None) and (end_date_pred != None) and (open_close != None) and (days_to_pred != None):
        if open_close==0: 
            target_dash = 'Open'
        else: target_dash = 'Close'
        #data conversion
        newstartdate = start_date_pred[:10]
        newenddate=end_date_pred[:10]
        a = datetime.strptime(str(newstartdate), '%Y-%m-%d')
        b = datetime.strptime(str(newenddate), '%Y-%m-%d')
        delta = b - a
        n_days = delta.days
        print(n_days)
        df_coin = fb.df_converter(df_coin, df_sp500, df_dollar)        
        fig2,rmse,mae,rsquare = fb.predictions(df_coin =df_coin, forecast_lenght = days_to_pred,train_lenght = int(n_days),target=target_dash)
        fig2.update_layout(
            template="plotly_dark",
            plot_bgcolor = 'rgba(0, 0, 0, 0)',
            paper_bgcolor = 'rgba(0, 0, 0, 0)',
            font_color="white",
            font_size= 15
        )
        return fig2, str(rmse), str(mae), str(rsquare)
    else: 
        print('waiting for all info to pred')
        fig2 = go.Figure(px.pie(values=[0],
                                names = ['']))
        fig2.update_layout(
            template="plotly_dark",
            plot_bgcolor = 'rgba(0, 0, 0, 0)',
            paper_bgcolor = 'rgba(0, 0, 0, 0)',
            font_color="white",
            font_size= 15
        )
        rmse = ''
        mae = ''
        rsquare = ''
        return fig2, rmse, mae,rsquare


###################################################################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
