import streamlit as st
import plotly.graph_objs as go
from datetime import datetime
import yfinance as yf
import pandas_ta as ta

def fetch_historical_data(product, period, interval):
    # Fetch historical data using yfinance
    data = yf.download(product, interval=interval, period=period)
    # Calculate EMAs using pandas_ta
    data['EMA_40'] = ta.ema(data['Close'], length=40)
    data['EMA_89'] = ta.ema(data['Close'], length=89)
    return data

# Sidebar
st.sidebar.title("Candlestick Charts")

# Select product (Forex, CFD, or Crypto)
product = st.sidebar.selectbox("Select Financial Instrument", [
    'GRANULES.NS','TATACOMM.NS','SRF.NS','SAIL.NS','TRENT.NS','MARUTI.NS','JSWSTEEL.NS',
    'JUBLFOOD.NS','GAIL.NS','HDFCLIFE.NS','POLYCAB.NS','ASIANPAINT.NS','LUPIN.NS',
    'ICICIGI.NS','GLENMARK.NS','SUNTV.NS','MFSL.NS','SYNGENE.NS','OBEROIRLTY.NS',
    'ATUL.NS','MCX.NS','ICICIPRULI.NS','NTPC.NS','TATAPOWER.NS','TORNTPHARM.NS',
    'AUROPHARMA.NS','DLF.NS','IEX.NS','ZYDUSLIFE.NS','ALKEM.NS','HINDALCO.NS',
    'BALKRISIND.NS','MRF.NS','NATIONALUM.NS','SBILIFE.NS','ABCAPITAL.NS','TVSMOTOR.NS',
    'ADANIPORTS.NS','GMRINFRA.NS','SIEMENS.NS','BAJAJ-AUTO.NS','HAVELLS.NS',
    'BHARTIARTL.NS','GODREJPROP.NS','HINDCOPPER.NS','MUTHOOTFIN.NS','SHREECEM.NS',
    'PEL.NS','PETRONET.NS','ADANIENT.NS','ASHOKLEY.NS','HEROMOTOCO.NS','CUMMINSIND.NS',
    'ABFRL.NS','KOTAKBANK.NS','VEDL.NS','JINDALSTEL.NS','IPCALAB.NS','MOTHERSON.NS',
    'ONGC.NS','UPL.NS','DIXON.NS','AARTIIND.NS','BERGEPAINT.NS','LTTS.NS',
    'PERSISTENT.NS','CIPLA.NS','COLPAL.NS','CROMPTON.NS','PIIND.NS','AMBUJACEM.NS',
    'PIDILITIND.NS','ITC.NS','COALINDIA.NS','LALPATHLAB.NS','TATASTEEL.NS','GUJGASLTD.NS',
    'SUNPHARMA.NS','EICHERMOT.NS','METROPOLIS.NS','LAURUSLABS.NS','ABBOTINDIA.NS',
    'TECHM.NS','TATACHEM.NS','ABB.NS','BIOCON.NS','SBICARD.NS','HCLTECH.NS',
    'BATAINDIA.NS','ICICIBANK.NS','LT.NS','ULTRACEMCO.NS','BPCL.NS','DEEPAKNTR.NS',
    'GODREJCP.NS','TCS.NS','COROMANDEL.NS','SHRIRAMFIN.NS','IDEA.NS','NAUKRI.NS',
    'BALRAMCHIN.NS','DABUR.NS','ESCORTS.NS','CHAMBLFERT.NS','NESTLEIND.NS',
    'HINDUNILVR.NS','NAVINFLUOR.NS','BAJAJFINSV.NS','BOSCHLTD.NS','ACC.NS',
    'CHOLAFIN.NS','DIVISLAB.NS','INDUSINDBK.NS','SBIN.NS','HDFCAMC.NS','MANAPPURAM.NS',
    'RAMCOCEM.NS','VOLTAS.NS','RECLTD.NS','IDFC.NS','INDIGO.NS','LTF.NS','PFC.NS',
    'UNITDSPR.NS','COFORGE.NS','GNFC.NS','HDFCBANK.NS','IDFCFIRSTB.NS','ASTRAL.NS',
    'FEDERALBNK.NS','NMDC.NS','BAJFINANCE.NS','TITAN.NS','IRCTC.NS','DALBHARAT.NS',
    'RELIANCE.NS','CUB.NS','LTIM.NS','TATAMOTORS.NS','AXISBANK.NS','M&M.NS',
    'APOLLOTYRE.NS','INFY.NS','INDHOTEL.NS','PAGEIND.NS','CANFINHOME.NS','POWERGRID.NS',
    'BEL.NS','HAL.NS','BHEL.NS','APOLLOHOSP.NS','BANDHANBNK.NS','M&MFIN.NS',
    'LICHSGFIN.NS','BANKBARODA.NS','UBL.NS','IGL.NS','GRASIM.NS','MPHASIS.NS',
    'BHARATFORG.NS','DRREDDY.NS','IOC.NS','TATACONSUM.NS','HINDPETRO.NS','MARICO.NS',
    'OFSS.NS','CONCOR.NS','RBLBANK.NS','BRITANNIA.NS','AUBANK.NS','INDIACEM.NS',
    'CANBK.NS','PVRINOX.NS','MGL.NS','EXIDEIND.NS','PNB.NS','JKCEMENT.NS',
    'INDUSTOWER.NS','BSOFT.NS','INDIAMART.NS'])

period = st.sidebar.selectbox("Select Period", ("1d","5d","1mo","max"))
interval = st.sidebar.selectbox("Select Interval", ("1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"))

# Fetch historical data
historical_data = fetch_historical_data(product, period, interval)

# Display candlestick chart
if historical_data is not None:
    # Extract data for plotting
    dates = historical_data.index.tolist()
    open_prices = historical_data['Open'].tolist()
    high_prices = historical_data['High'].tolist()
    low_prices = historical_data['Low'].tolist()
    close_prices = historical_data['Close'].tolist()
    ema_40 = historical_data['EMA_40'].tolist()
    ema_89 = historical_data['EMA_89'].tolist()

    # Detect crossovers
    cross_points_x = []
    cross_points_y = []
    for i in range(1, len(ema_40)):
        if (ema_40[i] > ema_89[i] and ema_40[i-1] < ema_89[i-1]) or (ema_40[i] < ema_89[i] and ema_40[i-1] > ema_89[i-1]):
            cross_points_x.append(dates[i])
            cross_points_y.append(ema_40[i])

    # Create candlestick chart using Plotly
    candlestick = go.Candlestick(x=dates,
                                 open=open_prices,
                                 high=high_prices,
                                 low=low_prices,
                                 close=close_prices)

    # Add EMA lines to the chart
    ema_40_line = go.Scatter(x=dates, y=ema_40, mode='lines', name='EMA 40', line=dict(color='#008000'))
    ema_89_line = go.Scatter(x=dates, y=ema_89, mode='lines', name='EMA 89', line=dict(color='#FF4040'))

    # Add plus symbols at crossover points
    cross_points = go.Scatter(x=cross_points_x, y=cross_points_y, mode='markers', name='Crossover',
                              marker=dict(symbol='cross', color='#239ED0', size=10))

    layout = go.Layout(
        title=f'Candlestick Chart for {product}',
        xaxis=dict(
            title='Date',
            type='category',
            rangebreaks=[
                dict(bounds=["15:15", "09:15"]),
                dict(values=["2024-07-15 15:15:00", "2024-07-16 09:15:00"]),
            ],
            tickformat='%H:%M',
            fixedrange=False
        ),
        yaxis=dict(
            title='Price',
            fixedrange=False  # Allows zooming in and out on the y-axis
        ),
        width=1000,   # Adjust width to fit Streamlit layout
        height=600,
        dragmode='zoom',  # Enable drag-to-zoom
        hovermode='x' 
    )

    fig = go.Figure(data=[candlestick, ema_40_line, ema_89_line, cross_points], layout=layout)
    # Display the chart using Streamlit
    st.plotly_chart(fig)
else:
    st.error("Failed to fetch historical data. Please check your API key and selected instrument.")
