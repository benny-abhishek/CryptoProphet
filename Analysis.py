import streamlit as st
import yfinance as yf
import pandas as pd
import datetime 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import base64

def app():
    
    @st.cache_data
    def locknload():
        s_coins = ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD']
        for i in range(0,len(s_coins)):
            coin = s_coins[i]
            if i==0:
                btc = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                continue
            if i==1:
                eth = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                continue
            if i==2:
                usdt = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
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
    
    #page layout
    st.set_page_config(layout="wide")
    mainpage_bg = '''<style>
    [data-testid="stAppViewContainer"]>.main{{
    background-image:url("image/img_file.jpg");
    background-size : cover;
    background-position : top left;
    background-repeat : no-repeat;
    backgorund-attachment:local;}}
    [data-testid="stHeader"]
    {{background:rgba(0,0,0,0);
    }}
    [data-testid="stToolbar"]
    {{right: 2rem;}}
    </style>'''
    st.markdown(mainpage_bg,unsafe_allow_html=True)
    #Title
    st.title("Crypto Prediction")

    #About at last
    col1=st.sidebar
    col1.header("Crypto Coins")

    currency_unit = "USD"

    s_coins = ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD']
    coin = col1.radio("Select a coin",('BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD'))

    btc,eth,usdt,bnb,usdc,xrp,ada,matic,doge,busd=locknload()

    st.subheader("Sample Price Data of "+coin)
    
    
    if coin =='BTC-USD':
        df=btc.copy()
    if coin =='ETH-USD':
        df=eth.copy()
    if coin =='USDT-USD':
        df=usdt.copy()
    if coin =='BNB-USD':
        df=bnb.copy()
    if coin =='USDC-USD':
        df=usdc.copy()
    if coin =='XRP-USD':
        df=xrp.copy()
    if coin =='ADA-USD':
        df=ada.copy()
    if coin =='MATIC-USD':
        df=matic.copy()
    if coin =='DOGE-USD':
        df=doge.copy()
    if coin =='BUSD-USD':
        df=busd.copy()
    
    st.table(df.tail(5))

    def download_file(df):
        csv=df.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
        return href
    st.markdown(download_file(df),unsafe_allow_html=True)

    figure=go.Figure()

    figure.add_trace(go.Scatter(x=df.index,y=df['Close'],name='Close'))

    figure.add_trace(go.Scatter(x=df.index,y=df['Open'],name='Open'))

    figure.update_layout(title='Opening and Closing Prices',yaxis_title='Crypto Price (USD)',height=700,width=1300)

    st.subheader('Opening and Closing Prices Of '+str(coin))
    
    st.plotly_chart(figure)
    #Low and high
    figure_LH=go.Figure()

    figure_LH.add_trace(go.Scatter(x=df.index,y=df['Low'],name='Low'))
    figure_LH.add_trace(go.Scatter(x=df.index,y=df['High'],name='High'))

    figure_LH.update_layout(title='High and Low Prices',yaxis_title='Crypto Price (USD)',height=700,width=1300)

    st.subheader('High and Low Prices Of '+str(coin))
    
    st.plotly_chart(figure_LH)

    #yayyy

    figure_vol=go.Figure()

    figure_vol.add_trace(go.Scatter(x=df.index,y=df['Volume'],name='Volume'))

    figure_vol.update_layout(title='Volume Sold',yaxis_title='Crypto Price (USD)',height=700,width=1300)

    st.subheader('Volume of '+str(coin)+' sold')
    st.plotly_chart(figure_vol)

    #st.markdown("""<iframe title="Crytocurrency y updated" width="930" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=b1ae4dd0-0031-49d4-8939-3fd411b967d5&autoAuth=true&ctid=83f3dabc-6f25-4958-bcab-9c1cd2b7fb2e" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)

app()
