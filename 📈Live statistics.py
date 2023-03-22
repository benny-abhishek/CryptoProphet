import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import datetime

def app():
    
    def live_price():
        s_coins = ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD']
        for i in range(0,len(s_coins)):
            coin=s_coins[i]

            if i==0:
                btc=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='1m')
                continue
            if i==1:
                eth=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='1m')
                continue
            if i==2:
                usdt=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='1m')
                continue
            if i==3:
                bnb=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='1m')
                continue
            if i==4:
                usdc=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='1m')
                continue
            if i==5:
                xrp=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='1m')
                continue
            if i==6:
                ada=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='1m')
                continue
            if i==7:
                matic=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='1m')
                continue
            if i==8:
                doge=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='1m')
                continue
            if i==9:
                busd = yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='1m')
                continue


        return btc,eth,usdt,bnb,usdc,xrp,ada,matic,doge,busd

    def live_data():
        s_coins = ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD']
        for i in range(0,len(s_coins)):
            coin=s_coins[i]

            if i==0:
                btc=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='15m')
                continue
            if i==1:
                eth=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='15m')
                continue
            if i==2:
                usdt=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='15m')
                continue
            if i==3:
                bnb=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='15m')
                continue
            if i==4:
                usdc=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='15m')
                continue
            if i==5:
                xrp=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='15m')
                continue
            if i==6:
                ada=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='15m')
                continue
            if i==7:
                matic=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='15m')
                continue
            if i==8:
                doge=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='15m')
                continue
            if i==9:
                busd=yf.download(tickers=coin,start=datetime.datetime.now()-datetime.timedelta(days=1),end=datetime.datetime.now(),interval='15m')
                continue

        return btc,eth,usdt,bnb,usdc,xrp,ada,matic,doge,busd
    
    st.title("Crypto Prediction")
    #expander_bar=st.expander("About")

    col1=st.sidebar
    col1.header("Crypto Coins")

    currency_unit = "USD"

    s_coins = ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD']
    coin = col1.radio("Select a coin",('BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD'))

    live_btc,live_eth,live_usdt,live_bnb,live_usdc,live_xrp,live_ada,live_matic,live_doge,live_busd=live_price()
    btc,eth,usdt,bnb,usdc,xrp,ada,matic,doge,busd=live_data()

    if coin =='BTC-USD':
        df=btc.copy()
        live_df=live_btc.copy()
    if coin =='ETH-USD':
        df=eth.copy()
        live_df=live_eth.copy()
    if coin =='USDT-USD':
        df=usdt.copy()
        live_df=live_usdt.copy()
    if coin =='BNB-USD':
        df=bnb.copy()
        live_df=live_bnb.copy()
    if coin =='USDC-USD':
        df=usdc.copy()
        live_df=live_usdc.copy()
    if coin =='XRP-USD':
        df=xrp.copy()
        live_df=live_xrp.copy()
    if coin =='ADA-USD':
        df=ada.copy()
        live_df=live_ada.copy()
    if coin =='MATIC-USD':
        df=matic.copy()
        live_df=live_matic.copy()
    if coin =='DOGE-USD':
        df=doge.copy()
        live_df=live_doge.copy()
    if coin =='BUSD-USD':
        df=busd.copy()
        live_df=live_busd.copy()
    
    st.subheader("Live Prices of Crypto Currencies")

    st.table(live_df.tail(5 ))

    figure=go.Figure()
    figure.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],name='Market Data'))

    figure.update_xaxes(rangeslider_visible=False,rangeselector=dict(buttons=list([dict(count=15,label='15m',step='minute',stepmode='backward'),
                                                                                    dict(count=45,label='45m',step='minute',stepmode='backward'),
                                                                                    dict(count=6,label='6h',step='hour',stepmode='todate'),
                                                                                    dict(step='all'),])))
    st.plotly_chart(figure)


app()


    


