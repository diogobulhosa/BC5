import pandas as pd 
import numpy as np 
import talib as tb
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import datetime
from datetime import date
from datetime import datetime as dt
from dateutil.relativedelta import *
from datetime import timedelta

from xgboost import XGBRegressor



def df_converter(df): 
    df_sp500 = yf.download('^GSPC', 
                      progress=False)
    df_dollar = yf.download('DX=F',  
                      progress=False)
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

    


