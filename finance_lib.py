import pandas as pd 
import numpy as np 
import talib as tb
import yfinance as yf

def df_converter(df): 
    df_sp500 = yf.download('^GSPC', 
                      start='2017-11-09', 
                      end='2022-05-08', 
                      progress=False)
    df_dollar = yf.download('DX=F', 
                      start='2017-11-09', 
                      end='2022-05-08', 
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