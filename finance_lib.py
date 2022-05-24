import pandas as pd 
import numpy as np 
import talib as tb
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from plotly.subplots import make_subplots
import datetime
from datetime import date
from datetime import datetime as dt
from dateutil.relativedelta import *
from datetime import timedelta

from xgboost import XGBRegressor






def df_converter(df, df_sp500, df_dollar): 
    # removed sp50 & dollar testing
    # clearing dollar and sp500 df
    df_dollar.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
    df_dollar.rename(columns={"Close": "dollar_close"}, inplace=True)
    df_sp500.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
    df_sp500.rename(columns={"Close": "sp500_close"}, inplace=True)
    # clearing general df
    #df_eth.drop('Unnamed: 0', axis=1, inplace=True)
    #df.drop('adj_close', axis=1, inplace=True)
    df.index = df.index.astype('datetime64[ns]')
    # MA df
    df_ma = df['Close'].to_frame()
    df_ma['SMA30'] = df_ma['Close'].rolling(15).mean()
    df_ma['CMA30'] = df_ma['Close'].expanding().mean()
    df_ma['EMA30'] = tb.EMA(df_ma['Close'], timeperiod=15)
    df_ma.dropna(inplace=True)
    # Stoch df
    slowk, slowd = tb.STOCH(df["High"], df["Low"], df["Close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df_stoch = pd.DataFrame(index=df.index,
                                data={"slowk": slowk,
                                    "slowd": slowd})
    df_stoch.dropna(inplace=True)
    # for later use in the concat
    stoch_c = ['slowk', 'slowd']
    # MACD df 
    macd, macdsignal, macdhist = tb.MACD(df.Close, fastperiod=12, slowperiod=26, signalperiod=9)
    df_macd = pd.DataFrame(index=df.index,
                            data={"macd": macd,
                                  "macdsignal": macdsignal,
                                  "macdhist": macdhist})
    df_macd.dropna(inplace=True)
    # for later use in the concat
    macd_c = ['macd', 'macdsignal', 'macdhist']
    # bb df
    upper, middle, lower = tb.BBANDS(df["Close"], timeperiod=15)
    df_bands = pd.DataFrame(index=df.index,
                                data={"bb_low": lower,
                                    "bb_ma": middle,
                                    "bb_high": upper})
    df_bands.dropna(inplace=True)
    # for later use in the concat
    bands_c = ['bb_low', 'bb_ma', 'bb_high']
    # rsi df
    rsi = tb.RSI(df['Close'], timeperiod=15)
    df_rsi = pd.DataFrame(index=df.index,
                            data={"close": df['Close'],
                                  "rsi": rsi})

    df_rsi.dropna(inplace=True)
    #stdev df
    stdev = tb.STDDEV(df['Close'], timeperiod=15, nbdev=1)
    df_stdev = pd.DataFrame(index=df.index,
                            data={"close": df['Close'],
                                  "stdev": stdev})
    df_stdev.dropna(inplace=True)
    # adx df
    adx = tb.ADX(df['High'], df['Low'], df['Close'], timeperiod=15)
    df_adx = pd.DataFrame(index=df.index,
                                data={"close": df['Close'],
                                    "adx": adx})

    df_adx.dropna(inplace=True)

    # concat 
    result =pd.concat([df, df_ma[['SMA30','CMA30','EMA30']], df_adx['adx'], df_bands[bands_c], df_macd[macd_c], df_rsi['rsi'], df_stdev['stdev'], df_stoch[stoch_c], df_dollar['dollar_close'], df_sp500['sp500_close']], axis=1)
    result.fillna(method='ffill', inplace=True)
    result.dropna(inplace=True)

    return result 

# to predict models functions
def add_days(df, forecast_length):
    end_point = len(df)
    df1 = pd.DataFrame(index=range(forecast_length), columns=range(2))
    df1.columns = ['Close', 'Date']
    df = df.append(df1)
    df = df.reset_index(drop=True)
    x = df.at[end_point - 1, 'Date']
    x = pd.to_datetime(x, format='%Y-%m-%d')
    for i in range(forecast_length):
        df.at[df.index[end_point + i], 'Date'] = x + timedelta(days=1+i)
        df.at[df.index[end_point + i], 'Close'] = 0
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df.drop(['Date'], axis=1)
    return df

def forecasting(model,df1, forecast_length):
    df3 = df1[['Close', 'Date']]
    df3 = add_days(df3, forecast_length)
    finaldf = df1.drop('Date', axis=1)
    finaldf = finaldf.reset_index(drop=True)
    end_point = len(finaldf)
    x = end_point - forecast_length
    finaldf_train = finaldf.loc[:x - 1, :]
    finaldf_train_x = finaldf_train.loc[:, finaldf_train.columns != 'Close']
    finaldf_train_y = finaldf_train['Close']

    fit = model.fit(finaldf_train_x, finaldf_train_y)
    yhat = []
    end_point = len(finaldf)
    df3_end = len(df3)
    for i in range(forecast_length, 0, -1):
        y = end_point - i
        inputfile = finaldf.loc[y:end_point, :]
        inputfile_x = inputfile.loc[:, inputfile.columns != 'Close']
        pred_set = inputfile_x.head(1)
        pred = fit.predict(pred_set)
        df3.at[df3.index[df3_end - i], 'Close'] = pred[0]
        finaldf = df1.drop('Date', axis=1)
        finaldf = finaldf.reset_index(drop=True)
        yhat.append(pred)
    yhat = np.array(yhat)
    return yhat

def predictions(df_coin,model, forecast_lenght = 5, train_lenght = 100,target = 'Close'):
    """ df_coin must be with date in index,
        forecast_lenght is the amount of days that we will predict
        model is the model predefined to use to get our predictions
        train_length is the amount of days that we will use to train the model
        target is what we are predicting
        This will return a graphic that will contain the data from the train set and our predictions
    """
    df_coin = df_coin.tail(train_lenght)
    df_coin.reset_index(inplace=True)
    df_coin['Date'] = pd.to_datetime(df_coin['Date'], format='%Y-%m-%d')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_coin['Date'], y=df_coin[target], 
                    name='Actual Values', mode='lines',line=dict(color='black')))
    #Predictions
    forecast = forecasting(model,df_coin,forecast_lenght)
    #df that will contain the predictions
    df_pred = pd.DataFrame(columns=['Date',target])
    #Adding the predictions to our dataset
    for day, x in enumerate(forecast):
        new_row={'Date':df_coin['Date'].max() + timedelta(days=1+day),
     target:x[0]}
        df_pred = df_pred.append(new_row, ignore_index=True)

    df_pred['Date'] = pd.to_datetime(df_pred['Date'], format='%Y-%m-%d')

    fig2.add_trace(go.Scatter(x=df_pred['Date'], y=df_pred[target], name='Predictions', mode='lines',line=dict(color='red')))
    fig2.update_layout(dict(updatemenus=[
                        dict(
                        type = "buttons",
                        direction = "left",
                        buttons=list([
                                dict(
                                args=["visible", "legendonly"],
                                label="Deselect All",
                                method="restyle"
                                ),
                                dict(
                                args=["visible", True],
                                label="Select All",
                                method="restyle"
                                )
                        ]),
                        pad={"r": 10, "t": 10},
                        showactive=False,
                        x=1,
                        xanchor="right",
                        y=1.1,
                        yanchor="top"
                        ),
                ]
        ))
    return fig2


def choose_model(model): 
   if model == 'XGB': 
     return  XGBRegressor(random_state = 10 , booster= 'gblinear', n_estimators = 130, validate_parameters = False,disable_default_eval_metric=False,eta = 0.3)


def list_converter(indicators): 
    for i,indicator in enumerate(indicators):
        if indicator == 'S&P 500': 
            indicators[i] = 'sp500_close'
        elif indicator == 'Dollar': 
            indicators[i] = 'dollar_close'
        elif indicator == 'STDEV': 
            indicators[i] = 'stdev'
    return indicators  

def candlestick(df, days, comparison = None, indicators = None):

    today = date.today()
    if indicators:
        indicators = list_converter(indicators)

    df = df[today - pd.offsets.Day(days):]
    df_normal = df.copy() # df with absolute values
    df= df.pct_change()

    color_palete= ['violet', 'magenta', 'turquoise', 'mediumorchid','lightpink', 'mediumpurple', 'mediumvioletred', 'darkviolet']

    #Editing the text(hover) on the candlestick to get the absolute value and the % changes   
    hovertext=[]
    for i in range(len(df_normal.Open)):
        hovertext.append('Open: '+str(df_normal.Open[i].round(2))+'<br>% change:' + str(df.Open[i].round(3))
                        +'<br>High: '+str(df_normal.High[i].round(2)) +'<br>% change:' + str(df.High[i].round(3))
                        +'<br>Low: '+str(df_normal.Low[i].round(2)) +'<br>% change:' + str(df.Low[i].round(3))
                        +'<br>Close: '+str(df_normal.Close[i].round(2)) +'<br>% change:' + str(df.Close[i].round(3)))

    if comparison is None and indicators is None: 

        fig = go.Figure(
            data = [
                go.Candlestick(
                    x = df_normal.index,
                    open = df_normal.Open,
                    high = df_normal.High,
                    low = df_normal.Low,
                    close = df_normal.Close,
                    text= hovertext,
                    hoverinfo='text'
                )
            ]
        )
        
        fig.update_layout(width=1000, height= 500)

    elif comparison is not None and indicators is None: 

        comparison = comparison[today - pd.offsets.Day(days):]
        comparison = comparison.pct_change()

        fig = go.Figure(
            data = [
                go.Candlestick(
                    x = df.index,
                    open = df.Open,
                    high = df.High,
                    low = df.Low,
                    close = df.Close,
                    text= hovertext,
                    hoverinfo='text'
                ),
                go.Scatter(
                    x = comparison.index, 
                    y = comparison.Close,
                    mode = 'lines', 
                    name = 'Extra Coin',
                    line = {'color': 'orange'},
                    customdata=df_normal[['Open','High','Low','Close']],
                            hovertemplate='<br>'.join([
                                'Percentual Change: %{y}',
                                'Date: %{x}',
                                'Open: %{customdata[0]:.3f}',
                                'High: %{customdata[1]:.3f}',
                                'Low: %{customdata[2]:.3f}',
                                'Close: %{customdata[3]:.3f}'
                            ])
                )

            ],
            layout = go.Layout(yaxis=dict(tickformat=".2%"))
        )   
        # side when you add coin
        fig.update_layout(width=1000, height= 800)

    elif comparison is None and indicators is not None:

        fig = make_subplots(rows=5, cols=1, specs = [[{ }], [{ }], [{ }], [{ }], [{ }]], vertical_spacing = 0.10, 
                            row_heights=[180, 30, 30, 30, 30])
        fig.update_yaxes(tickformat=".2%", row=1)
        fig.append_trace(
                        go.Candlestick(
                            x = df.index,
                            open = df.Open,
                            high = df.High,
                            low = df.Low,
                            close = df.Close,
                            text= hovertext,
                            hoverinfo='text'
                        ), row=1, col=1)
        
        for i,v in enumerate(indicators): 
            if v == 'RSI': 
                fig.append_trace(go.Scatter(
                        x=df_normal['rsi'].index,
                        y=df_normal['rsi'],
                        mode="lines",
                        name = 'rsi'
                        ),  row=3, col=1
                    )
                
                fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.5, showarrow=False,
                   text="<b>RSI</b>", row=3, col=1)
                
                fig.add_hline(y=30 , line_width=2, line_dash="dash", line_color="darkslategray", row=3, col=1)
                fig.add_hline(y=70, line_width=2, line_dash="dash", line_color="darkslategray", row=3, col=1)

            
            elif v == 'ADX': 
                fig.append_trace(go.Scatter(
                        x=df_normal['adx'].index,
                        y=df_normal['adx'],
                        mode="lines",
                        name = 'adx'
                        ),  row=5, col=1
                    )
                
                fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.5, showarrow=False,
                   text="<b>ADX</b>", row=5, col=1)

            elif v == 'Stochastic Oscillator': 
                fig.append_trace(go.Scatter(
                        x=df_normal['slowk'].index,
                        y=df_normal['slowk'],
                        mode="lines",
                        name = 'slowk'
                        ),  row=2, col=1
                    )
                fig.add_trace(
                        go.Scatter(
                            x=df_normal['slowd'].index,
                            y=df_normal['slowd'],
                            mode="lines",
                            name = 'slowd'
                            ),  row=2, col=1
                    )

                fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.5, showarrow=False,
                   text="<b>Stochastic Oscillator</b>", row=2, col=1)
                fig.add_hline(y=20 , line_width=2, line_dash="dash", line_color="darkslategray", row=2, col=1)
                fig.add_hline(y=80, line_width=2, line_dash="dash", line_color="darkslategray", row=2, col=1)

            elif v == 'MACD': 
                fig.append_trace(go.Scatter(
                        x=df_normal['macd'].index,
                        y=df_normal['macd'],
                        mode="lines",
                        name = 'macd'
                        ),  row=4, col=1
                    )
                fig.add_trace(
                        go.Scatter(
                            x=df_normal['macdsignal'].index,
                            y=df_normal['macdsignal'],
                            mode="lines",
                            name = 'macd signal'
                            ),  row=4, col=1
                    )
                
                fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.5, showarrow=False,
                   text="<b>MACD</b>", row=4, col=1)
                          
            elif v == 'Bollinger Bands':

                fig.add_trace(go.Scatter(x=df['bb_low'].index,
                    y=df['bb_low'],
                    fill=None,
                    mode='lines',
                    line_color='indigo',
                    name = 'bb_low',
                    customdata=df_normal[['Open','High','Low','Close']],
                            hovertemplate='<br>'.join([
                                'Percentual Change: %{y}',
                                'Date: %{x}',
                                'Open: %{customdata[0]:.3f}',
                                'High: %{customdata[1]:.3f}',
                                'Low: %{customdata[2]:.3f}',
                                'Close: %{customdata[3]:.3f}'
                            ])
                    
                    ))
                fig.add_trace(go.Scatter(
                    x=df['bb_high'].index,
                    y=df['bb_high'],
                    name = 'bb_high',
                    fill='tonexty', # fill area between trace0 and trace1
                    mode='lines', line_color='indigo', 
                    customdata=df_normal[['Open','High','Low','Close']],
                            hovertemplate='<br>'.join([
                                'Percentual Change: %{y}',
                                'Date: %{x}',
                                'Open: %{customdata[0]:.3f}',
                                'High: %{customdata[1]:.3f}',
                                'Low: %{customdata[2]:.3f}',
                                'Close: %{customdata[3]:.3f}'
                            ])))    

            else:
                for ind, value in enumerate(color_palete):
                    if i == ind: 
                        fig.add_trace(
                        go.Scatter(
                            x=df[v].index,
                            y=df[v],
                            mode="lines",
                            name = v,
                            line_color=value,
                            customdata=df_normal[['Open','High','Low','Close']],
                            hovertemplate='<br>'.join([
                                'Percentual Change: %{y}',
                                'Date: %{x}',
                                'Open: %{customdata[0]:.3f}',
                                'High: %{customdata[1]:.3f}',
                                'Low: %{customdata[2]:.3f}',
                                'Close: %{customdata[3]:.3f}'
                            ]),
                        )
                        )
                
        fig.update_layout(width=1000, height= 1300)
    
    else:

        comparison = comparison[today - pd.offsets.Day(days):]
        comparison = comparison.pct_change()
        
        fig = make_subplots(rows=5, cols=1, specs = [[{ }], [{ }], [{ }],  [{ }], [{ }]], vertical_spacing = 0.10, 
                            row_heights=[180, 30, 30, 30, 30] )
        fig.update_yaxes(tickformat=".2%", row = 1)

        fig.append_trace(
                go.Candlestick(
                    x = df.index,
                    open = df.Open,
                    high = df.High,
                    low = df.Low,
                    close = df.Close,
                    text= hovertext,
                    hoverinfo='text'
                ), row=1, col=1)
        
        
        
        fig.add_trace(go.Scatter(
                        x = comparison.index, 
                        y = comparison.Close,
                        mode = 'lines', 
                        name = 'Extra Coin',
                        line = {'color': 'orange'},
                        customdata=df_normal[['Open','High','Low','Close']],
                            hovertemplate='<br>'.join([
                                'Percentual Change: %{y}',
                                'Date: %{x}',
                                'Open: %{customdata[0]:.3f}',
                                'High: %{customdata[1]:.3f}',
                                'Low: %{customdata[2]:.3f}',
                                'Close: %{customdata[3]:.3f}'
                            ])
                        ))

        for i,v in enumerate(indicators): 
            if v == 'RSI': 
            
                fig.append_trace(go.Scatter(
                        x=df_normal['rsi'].index,
                        y=df_normal['rsi'],
                        mode="lines",
                        name = 'rsi'
                        ),  row=3, col=1
                    )
                
                fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.5, showarrow=False,
                   text="<b>RSI</b>", row=3, col=1)
                fig.add_hline(y=30 , line_width=2, line_dash="dash", line_color="darkslategray", row=3, col=1)
                fig.add_hline(y=70, line_width=2, line_dash="dash", line_color="darkslategray", row=3, col=1)

            elif v == 'ADX': 
                fig.append_trace(go.Scatter(
                        x=df_normal['adx'].index,
                        y=df_normal['adx'],
                        mode="lines",
                        name = 'adx'
                        ),  row=5, col=1
                    )
                
                fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.5, showarrow=False,
                   text="<b>ADX</b>", row=5, col=1)

            elif v == 'Stochastic Oscillator': 
                fig.append_trace(go.Scatter(
                        x=df_normal['slowk'].index,
                        y=df_normal['slowk'],
                        mode="lines",
                        name = 'slowk'
                        ),  row=2, col=1
                    )
                fig.add_trace(
                        go.Scatter(
                            x=df_normal['slowd'].index,
                            y=df_normal['slowd'],
                            mode="lines",
                            name = 'slowd'
                            ),  row=2, col=1
                    )

                fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.5, showarrow=False,
                   text="<b>Stochastic Oscillator</b>", row=2, col=1)
                fig.add_hline(y=20 , line_width=2, line_dash="dash", line_color="darkslategray", row=2, col=1)
                fig.add_hline(y=80, line_width=2, line_dash="dash", line_color="darkslategray", row=2, col=1)

            elif v == 'MACD': 
                fig.append_trace(go.Scatter(
                        x=df_normal['macd'].index,
                        y=df_normal['macd'],
                        mode="lines",
                        name = 'macd'
                        ),  row=4, col=1
                    )
                fig.add_trace(
                        go.Scatter(
                            x=df_normal['macdsignal'].index,
                            y=df_normal['macdsignal'],
                            mode="lines",
                            name = 'macd signal'
                            ),  row=4, col=1
                    )
                
                fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.5, showarrow=False,
                   text="<b>MACD</b>", row=4, col=1)
            
            elif v == 'Bollinger Bands':

                fig.add_trace(go.Scatter(x=df['bb_low'].index,
                    y=df['bb_low'],
                    fill=None,
                    mode='lines',
                    line_color='indigo',
                    name = 'bb_low',
                    customdata=df_normal[['Open','High','Low','Close']],
                            hovertemplate='<br>'.join([
                                'Percentual Change: %{y}',
                                'Date: %{x}',
                                'Open: %{customdata[0]:.3f}',
                                'High: %{customdata[1]:.3f}',
                                'Low: %{customdata[2]:.3f}',
                                'Close: %{customdata[3]:.3f}'
                            ])
                    
                    ))
                fig.add_trace(go.Scatter(
                    x=df['bb_high'].index,
                    y=df['bb_high'],
                    name = 'bb_high',
                    fill='tonexty', # fill area between trace0 and trace1
                    mode='lines', line_color='indigo', 
                    customdata=df_normal[['Open','High','Low','Close']],
                            hovertemplate='<br>'.join([
                                'Percentual Change: %{y}',
                                'Date: %{x}',
                                'Open: %{customdata[0]:.3f}',
                                'High: %{customdata[1]:.3f}',
                                'Low: %{customdata[2]:.3f}',
                                'Close: %{customdata[3]:.3f}'
                            ])))
                    
            else:
                
                for ind, value in enumerate(color_palete):

                    if i == ind: 
                        fig.add_trace(
                        go.Scatter(
                            x=df[v].index,
                            y=df[v],
                            mode="lines",
                            name = v,
                            line_color=value,
                            customdata=df_normal[['Open','High','Low','Close']],
                            hovertemplate='<br>'.join([
                                'Percentual Change: %{y}',
                                'Date: %{x}',
                                'Open: %{customdata[0]:.3f}',
                                'High: %{customdata[1]:.3f}',
                                'Low: %{customdata[2]:.3f}',
                                'Close: %{customdata[3]:.3f}'
                            ]),
                        )
                        )
        fig.update_layout(width=1000, height= 500)
                        
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        font_color="white",
        font_size= 15
    )       
    return fig