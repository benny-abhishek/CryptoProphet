import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta
import datetime 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import base64
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import psycopg2
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,LSTM,Dropout


def app():
    def load_data():
        s_coins = ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD']
        for i in range(0,len(s_coins)):
            coin = s_coins[i]
            if i==0:
                btc = yf.download(tickers=coin,start="2012-01-01",end=datetime.datetime.now())
                continue
            if i==1:
                eth = yf.download(tickers=coin,start="2012-01-01",end=datetime.datetime.now())
                continue
            if i==2:
                usdt = yf.download(tickers=coin,start="2012-01-01",end=datetime.datetime.now())
                continue
            if i==3:
                bnb = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                continue
            if i==4:
                usdc = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                continue
            if i==5:
                xrp = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                continue
            if i==6:
                ada = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                continue
            if i==7:
                matic = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                continue
            if i==8:
                doge = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                continue
            if i==9:
                busd = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                continue
        return btc,eth,usdt,bnb,usdc,xrp,ada,matic,doge,busd
    def load():
        #postgre sql connection code

        hostname = 'localhost'
        database = 'crypto'
        port_id = 5432
        username = 'postgres'
        pwd = 'secret123'
        conn=psycopg2.connect(
        host=hostname,
        port=port_id,
        dbname=database,
        user=username,
        password=pwd)
        
        #conn=None
        #cur=None
        column_names = ['Date','Open','High','Low','Close','Volume','Currency']
        conn.autocommit = True
        cur = conn.cursor()
        s_coins = ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD']
        for i in range(0,len(s_coins)):
            coin = s_coins[i]
            if i==0:
                cur = conn.cursor()
                cur.execute('''SELECT * FROM bitcoin;''')
                tuples=cur.fetchall()
                btc = pd.DataFrame(tuples, columns=column_names)
                continue
            if i==1:
                cur = conn.cursor()
                cur.execute('''SELECT * FROM ethereum;''')
                tuples=cur.fetchall()
                eth = pd.DataFrame(tuples, columns=column_names)
                continue
            if i==2:
                cur = conn.cursor()
                cur.execute('''SELECT * FROM tether;''')
                tuples=cur.fetchall()
                usdt = pd.DataFrame(tuples, columns=column_names)
                #usdt =pd.read_csv('tether.csv')
                continue
            if i==3:
                cur = conn.cursor()
                cur.execute('''SELECT * FROM bnb;''')
                tuples=cur.fetchall()
                bnb = pd.DataFrame(tuples, columns=column_names)
                #bnb = pd.read_csv('bnb.csv')
                continue
            if i==4:
                cur = conn.cursor()
                cur.execute('''SELECT * FROM usd_coin;''')
                tuples=cur.fetchall()
                usdc = pd.DataFrame(tuples, columns=column_names)
                #usdc = pd.read_csv('usd_coin.csv')
                continue
            if i==5:
                cur = conn.cursor()
                cur.execute('''SELECT * FROM xrp;''')
                tuples=cur.fetchall()
                xrp = pd.DataFrame(tuples, columns=column_names)
                #xrp = pd.read_csv('xrp.csv')
                continue
            if i==6:
                cur = conn.cursor()
                cur.execute('''SELECT * FROM cardano;''')
                tuples=cur.fetchall()
                ada = pd.DataFrame(tuples, columns=column_names)
                #ada = pd.read_csv('cardano.csv')
                continue
            if i==7:
                cur = conn.cursor()
                cur.execute('''SELECT * FROM polygon;''')
                tuples=cur.fetchall()
                matic = pd.DataFrame(tuples, columns=column_names)
                #matic = pd.read_csv('polygon.csv')
                continue
            if i==8:
                cur = conn.cursor()
                cur.execute('''SELECT * FROM dogecoin;''')
                tuples=cur.fetchall()
                doge = pd.DataFrame(tuples, columns=column_names)
                #doge = pd.read_csv('dogecoin.csv')
                continue
            if i==9:
                cur = conn.cursor()
                cur.execute('''SELECT * FROM Binance_USD;''')
                tuples=cur.fetchall()
                busd = pd.DataFrame(tuples, columns=column_names)
                #busd = pd.read_csv('Binance_USD.csv')
                continue
        cur.close()
        conn.close()
        return btc,eth,usdt,bnb,usdc,xrp,ada,matic,doge,busd
    btc,eth,usdt,bnb,usdc,xrp,ada,matic,doge,busd=load()
    btc_df,eth_df,usdt_df,bnb_df,usdc_df,xrp_df,ada_df,matic_df,doge_df,busd_df= load_data()
    st.set_page_config(layout="wide")

    #Title
    st.title("Crypto Prediction")

    #About at last
    col1=st.sidebar
    col1.header("Crypto Coins")

    currency_unit = "USD"

    s_coins = ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD']
    coin = col1.radio("Select a coin",('BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD'))
    
    
    if coin =='BTC-USD':
       df=btc.copy()
       dfs = btc_df.copy()

    if coin =='ETH-USD':
       df=eth.copy()
       dfs=eth_df.copy()
    if coin =='USDT-USD':
       df=usdt.copy()
       dfs=usdt_df.copy()

    if coin =='BNB-USD':
       df=bnb.copy()
       dfs=bnb_df.copy()

    if coin =='USDC-USD':
       df=usdc.copy()
       dfs=usdc_df.copy()

    if coin =='XRP-USD':
       df=xrp.copy()
       dfs=xrp_df.copy()

    if coin =='ADA-USD':
       df=ada.copy()
       dfs=ada_df.copy()

    if coin =='MATIC-USD':
       df=matic.copy()
       dfs=matic_df.copy()

    if coin =='DOGE-USD':
       df=doge.copy()
       dfs=doge_df.copy()

    if coin =='BUSD-USD':
       df=busd.copy()
       dfs=busd_df.copy()

    #SVR---------------------------------------------------------------------

    st.subheader("Support Vector Regression (SVR) for "+coin)

    df.dropna(inplace=True)
    df1 = df[['Close']]
    df1['Prediction'] = df1[['Close']].shift(-60)

    X = np.array(df1.drop(['Prediction'],1))
    X = X[:-60]

    y = np.array(df1['Prediction'])
    y = y[:-60]

    scaler = StandardScaler()
    X = scaler.fit_transform(X) 
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    svr = SVR(kernel='rbf', C=1000, gamma=0.1)
    svr.fit(x_train, y_train)
    #svm_confidence = svr.score(x_test, y_test)

    y_pred = svr.predict(x_test)
    dummy=np.zeros(1)
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=dummy))
    fig.add_trace(go.Scatter(y=y_test,name='Real Price'))
    fig.add_trace(go.Scatter(y=y_pred,name='Predicted Price'))
    fig.update_layout(title='SVR',yaxis_title='Crypto Price (USD)',height=700,width=1300)
    st.plotly_chart(fig)



    #linear----------------------------------------------------------------------

    st.subheader("Linear Regression for "+coin)
    #df2.dropna(inplace=True)
    df2=df.copy()
    
    df2.set_index(pd.DatetimeIndex(df['Date']), inplace=True)

    df2 = df2[['Close']]

    # Add EMA (Technical Indicators) to dataframe representing the exponential moving average calculated over a 10-day period
    df2.ta.ema(close='Close', length=10, append=True)
    df2.dropna(inplace=True)
    # Split data into testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(df2[['Close']], df2[['EMA_10']], test_size=0.2)

    # Create Regression Model
    model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.style.use('dark_background')
    fig=plt.figure(figsize=(7,3))
    plt.scatter(X_train, y_train, color = 'steelblue', label='Real values')
    plt.plot(X_train, model.predict(X_train), color = 'red', label='Predicted values')
    plt.legend()
    st.pyplot(fig)

    #decision tree--------------------------------------------------------------------------------------------
    st.subheader("Decision Tree Algorithm for "+coin)

    df3 = df[['Close']]

    #print(df2.head())
    # Prediction 100 days into the future
    future_days = 60
    df3['Prediction'] = df3[['Close']].shift(-future_days)
    X = np.array(df3.drop(['Prediction'], 1))[:-future_days]
    y = np.array(df3['Prediction'])[:-future_days]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    tree = DecisionTreeRegressor().fit(x_train, y_train)

    predictions = tree.predict(X)

    fig=plt.figure(figsize=(8,4))
    plt.xlabel('Days')
    plt.ylabel('Close Price USD ($)')
    plt.plot(X,color='steelblue')
    plt.plot(predictions,color='orange')
    plt.legend(['Original', 'Predicted'])
    #plt.savefig('plot.png')
    #st.image('plot.png')
    st.pyplot(fig)

    #LSTM-------------------------------------------------------------------------------------------
    st.subheader("LSTM Algorithm for "+coin)

    close_data = dfs['Close'].values
    close_data = close_data.reshape((-1,1))

    split_percent = 0.90
    split = int(split_percent*len(close_data))

    close_train = close_data[:split]
    close_test = close_data[split:]

    date_train = dfs.index[:split]
    date_test = dfs.index[split:]


    look_back = 15

    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)



    model = Sequential()
    model.add(LSTM(10,activation='relu',input_shape=(look_back,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    num_epochs = 25
    model.fit_generator(train_generator, epochs=50, verbose=1)
    prediction = model.predict(test_generator)

    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))

    trace1 = go.Scatter(x = date_train,y = close_train,mode = 'lines',name = 'Data')
    trace2 = go.Scatter(x = date_test,y = prediction,mode = 'lines',name = 'Prediction')
    trace3 = go.Scatter(x = date_test,y = close_test,mode='lines',name = 'Ground Truth')
    layout = go.Layout(title = f"{coin} Close Price Prediction",xaxis = {'title' : "Date"},yaxis = {'title' : "Close"},height=700,width=1300)
    fig = go.Figure(data=[trace1,trace2, trace3], layout=layout)

    st.plotly_chart(fig)

    

app()