import streamlit as st
import plotly.graph_objs as go
# from datetime import datetime
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta

import pandas as pd



def fetch_historical_data(product, period, interval):
    # Fetch historical data using yfinance
    data = yf.download(product, interval=interval, period=period)
    
    # Fetch stock data for dividend and earnings information
    stock = yf.Ticker(product)
    
    # Dividend data
    upcoming_dividends = stock.dividends
    next_dividend_date = None
    days_left_dividend = 0
    
    if not upcoming_dividends.empty:
        next_dividend_date = upcoming_dividends.index[-1].strftime('%Y-%m-%d')
        next_dividend_datetime = datetime.strptime(next_dividend_date, '%Y-%m-%d')
        today = datetime.today()
        if next_dividend_datetime > today:
            days_left_dividend = (next_dividend_datetime - today).days
        else:
            days_left_dividend = 0

    # Earnings data
    upcoming_earnings = [] #stock.earnings_dates
    next_earnings_date = None
    days_left_earnings = 0

    #if not upcoming_earnings.empty:
        #next_earnings_date = upcoming_earnings.index[-1].strftime('%Y-%m-%d')
        #next_earnings_datetime = datetime.strptime(next_earnings_date, '%Y-%m-%d')
        #if next_earnings_datetime > today:
           # days_left_earnings = (next_earnings_datetime - today).days
        #else:
            #days_left_earnings = 0

    # Quarterly results (assuming quarterly results are linked to earnings dates)
    next_quarterly_results_date = next_earnings_date
    days_left_quarterly_results = days_left_earnings

    # Calculate EMAs using pandas_ta
    data['EMA_40'] = ta.ema(data['Close'], length=40)
    data['EMA_89'] = ta.ema(data['Close'], length=89)
    
    return data, next_dividend_date, days_left_dividend, next_earnings_date, days_left_earnings, next_quarterly_results_date, days_left_quarterly_results

# Sidebar
st.sidebar.title("Candlestick Charts")
selected_page = st.sidebar.selectbox("Select Page", ["Chart", "Trend List","All Stock List","HEATMAP","HEATMAP Volume"])
period = st.sidebar.selectbox("Select Period", ("1d", "5d", "1mo", "max"))
interval = st.sidebar.selectbox("Select Interval", ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"))

stock_symbols = ['GRANULES.NS','TATACOMM.NS','SRF.NS','SAIL.NS','TRENT.NS','MARUTI.NS','JSWSTEEL.NS',
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
        'INDUSTOWER.NS','BSOFT.NS','INDIAMART.NS']

if selected_page == "Chart":
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

    
    # Fetch historical data
    historical_data, next_dividend_date, days_left_dividend, next_earnings_date, days_left_earnings, next_quarterly_results_date, days_left_quarterly_results = fetch_historical_data(product, period, interval)

    # Display upcoming dates in the sidebar
    if next_dividend_date and days_left_dividend > 0:
        st.sidebar.success(f"Next Dividend Date: {next_dividend_date}")
        st.sidebar.info(f"Days Left Until Dividend: {days_left_dividend} days")
    else:
        st.sidebar.warning("No upcoming dividend date found.")

    if next_earnings_date and days_left_earnings > 0:
        st.sidebar.success(f"Next Earnings Date: {next_earnings_date}")
        st.sidebar.info(f"Days Left Until Earnings: {days_left_earnings} days")
    else:
        st.sidebar.warning("No upcoming earnings date found.")

    if next_quarterly_results_date and days_left_quarterly_results > 0:
        st.sidebar.success(f"Next Quarterly Results Date: {next_quarterly_results_date}")
        st.sidebar.info(f"Days Left Until Quarterly Results: {days_left_quarterly_results} days")
    else:
        st.sidebar.warning("No upcoming quarterly results date found.")

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

elif selected_page == "Trend List":
    # Add code for the second page here
    
    # stock_symbols = ['TATACOMM.NS','ADANIENT.NS']
# Function to fetch and calculate EMAs
    def get_ema_crossovers(symbol):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Get data for the last day
        data = yf.download(symbol, interval=interval, period=period)
        
    
        data['EMA_40'] = ta.ema(data['Close'], length=40)  # Use pandas_ta to calculate SMA
        data['EMA_89'] = ta.ema(data['Close'], length=89)
        checkshot = data.ta.supertrend(length=10, multiplier=3, append=True)
        # print(checkshot.iloc[-1])
        checkshort = 'SELL' if checkshot.iloc[-1]["SUPERTd_10_3.0"] == -1 else 'BUY'
        data ['TREND'] = checkshort
        data.drop(columns=["SUPERTd_10_3.0"], inplace=True)
        data.drop(columns=["SUPERTl_10_3.0"], inplace=True)
        data.drop(columns=["SUPERTs_10_3.0"], inplace=True)
        data.drop(columns=["SUPERT_10_3.0"], inplace=True)
        # Convert data.index to timezone-naive if necessary
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
    
        # Check if data contains enough rows
        if len(data) < 3:
            return None
    
        # Identify the time 15 minutes ago
        now = datetime.now()
        fifteen_minutes_ago = now - timedelta(minutes=15)
        fifteen_minutes_ago = fifteen_minutes_ago.replace(second=0, microsecond=0)
    
        # Find the closest time to 15 minutes ago
        closest_time = min(data.index, key=lambda t: abs(t - fifteen_minutes_ago))
    
        
        # Get the data for the closest time
        if closest_time not in data.index:
            return None
    
        index_position = data.index.get_loc(closest_time)
        # Ensure we have previous data to compare
        if index_position < 1:
            return None
    
        # Get the data for the closest time and the one before it
        last_two = data.iloc[index_position-1:index_position+1]
        if len(last_two) < 2:
            return None
    
        # Condition for a crossover
        percent_start = data.iloc[-1]['Close'] * 0.005
        # print(data)
        if ((last_two['EMA_40'].iloc[-1] <= last_two['EMA_89'].iloc[-1]) and (abs(last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1]) >= 0 ) and (abs(last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1] ) <= percent_start)):
            # print(last_two['EMA_40'].iloc[-1] ,' || ', last_two['EMA_89'].iloc[-1] ,' || ', last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1] ,' || ', symbol, ' || ', percent_start ,' || \n',last_two.tail(1))
            
            daa =daa = [symbol, last_two['Open'].iloc[-1], last_two['Close'].iloc[-1], last_two['High'].iloc[-1], last_two['Low'].iloc[-1], last_two['Adj Close'].iloc[-1], last_two['EMA_40'].iloc[-1], last_two['EMA_89'].iloc[-1], last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1] , last_two['TREND'].iloc[-1]]
            return daa
        # if (last_two['EMA_40'].iloc[-2] <= last_two['EMA_89'].iloc[-2]) and (last_two['EMA_40'].iloc[-1] > last_two['EMA_89'].iloc[-1]):
        #     return last_two.tail(1)
    
        return None
    
    # Check for buy signals across all stocks
    buy_signals = {}
    data = []
    for symbol in stock_symbols:
        signal = get_ema_crossovers(symbol)
        if signal is not None:
            buy_signals[symbol] = signal
    
    # Print buy signals
    for symbol, signal in buy_signals.items():
        # print(f"Buy signal for {symbol}:")
        # print(signal)
        da = signal
        data.append(da)
    
    if not buy_signals:
        st.sidebar.info("No buy signals found.")
        # ... add any other content for the second page ... 

    if buy_signals:
        data = sorted(data, key=lambda x: x[-2], reverse=True)
        columns = ['symbol', 'Open', 'Close', 'High', 'Low', 'Adj Close', 'EMA_89', 'EMA_40','EMA_40 - EMA_89', 'Trend']
        df = pd.DataFrame(data, columns=columns)
        st.table(df)
    else:
        st.sidebar.info("No buy signals found.")

elif selected_page == "All Stock List":
    # Add code for the second page here
    
    # stock_symbols = ['TATACOMM.NS','ADANIENT.NS']
# Function to fetch and calculate EMAs
    def get_ema_crossovers(symbol):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Get data for the last day
        data = yf.download(symbol, interval=interval, period=period)
        
    
        data['EMA_40'] = ta.ema(data['Close'], length=40)  # Use pandas_ta to calculate SMA
        data['EMA_89'] = ta.ema(data['Close'], length=89)
        checkshot = data.ta.supertrend(length=10, multiplier=3, append=True)
        # print(checkshot.iloc[-1])
        checkshort = 'SELL' if checkshot.iloc[-1]["SUPERTd_10_3.0"] == -1 else 'BUY'
        data ['TREND'] = checkshort
        data.drop(columns=["SUPERTd_10_3.0"], inplace=True)
        data.drop(columns=["SUPERTl_10_3.0"], inplace=True)
        data.drop(columns=["SUPERTs_10_3.0"], inplace=True)
        data.drop(columns=["SUPERT_10_3.0"], inplace=True)
        # Convert data.index to timezone-naive if necessary
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
    
        # Check if data contains enough rows
        if len(data) < 3:
            return None
    
        # Identify the time 15 minutes ago
        now = datetime.now()
        fifteen_minutes_ago = now - timedelta(minutes=15)
        fifteen_minutes_ago = fifteen_minutes_ago.replace(second=0, microsecond=0)
    
        # Find the closest time to 15 minutes ago
        closest_time = min(data.index, key=lambda t: abs(t - fifteen_minutes_ago))
    
        
        # Get the data for the closest time
        if closest_time not in data.index:
            return None
    
        index_position = data.index.get_loc(closest_time)
        # Ensure we have previous data to compare
        if index_position < 1:
            return None
    
        # Get the data for the closest time and the one before it
        last_two = data.iloc[index_position-1:index_position+1]
        if len(last_two) < 2:
            return None
    
        # Condition for a crossover
        percent_start = data.iloc[-1]['Close'] * 0.005
        # print(data)
        # if ((last_two['EMA_40'].iloc[-1] <= last_two['EMA_89'].iloc[-1]) and (abs(last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1]) >= 0 ) and (abs(last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1] ) <= percent_start)):
            # print(last_two['EMA_40'].iloc[-1] ,' || ', last_two['EMA_89'].iloc[-1] ,' || ', last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1] ,' || ', symbol, ' || ', percent_start ,' || \n',last_two.tail(1))
            
        daa = [symbol, last_two['Open'].iloc[-1], last_two['Close'].iloc[-1], last_two['High'].iloc[-1], last_two['Low'].iloc[-1], last_two['Adj Close'].iloc[-1], last_two['EMA_40'].iloc[-1], last_two['EMA_89'].iloc[-1], last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1] , last_two['TREND'].iloc[-1]]
        return daa
        # if (last_two['EMA_40'].iloc[-2] <= last_two['EMA_89'].iloc[-2]) and (last_two['EMA_40'].iloc[-1] > last_two['EMA_89'].iloc[-1]):
        #     return last_two.tail(1)
    
        # return None
    
    # Check for buy signals across all stocks
    buy_signals = {}
    data = []
    for symbol in stock_symbols:
        signal = get_ema_crossovers(symbol)
        if signal is not None:
            buy_signals[symbol] = signal
    
    # Print buy signals
    for symbol, signal in buy_signals.items():
        # print(f"Buy signal for {symbol}:")
        # print(signal)
        da = signal
        data.append(da)
    
    if not buy_signals:
        st.sidebar.info("No buy signals found.")
        # ... add any other content for the second page ... 

    if buy_signals:
        data = sorted(data, key=lambda x: x[-2], reverse=True)
        columns = ['symbol', 'Open', 'Close', 'High', 'Low', 'Adj Close', 'EMA_89', 'EMA_40','EMA_40 - EMA_89', 'Trend']
        df = pd.DataFrame(data, columns=columns)
        st.table(df)
    else:
        st.sidebar.info("No buy signals found.")

elif selected_page == "HEATMAP":
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    def get_ema_crossovers(symbol):
        data = yf.download(symbol, interval=interval, period=period)

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        if len(data) < 2:
            return None

        yesterday = datetime.now() - timedelta(days=1)
        yesterday_close_time = yesterday.replace(hour=15, minute=15, second=0, microsecond=0)

        closest_time_yesterday = min(data.index, key=lambda t: abs(t - yesterday_close_time))
        yesterday_close_price = data.loc[closest_time_yesterday]['Close']

        today_current_price = data.iloc[-1]['Close']

        price_diff = today_current_price - yesterday_close_price
        price_change_percent = (price_diff / yesterday_close_price) * 100

        return [symbol, price_change_percent, price_diff]

    buy_signals = []
    for symbol in stock_symbols:
        signal = get_ema_crossovers(symbol)
        if signal is not None:
            buy_signals.append(signal)

    if buy_signals:
        data = sorted(buy_signals, key=lambda x: x[-2], reverse=True)
        df = pd.DataFrame(data, columns=['Symbol', 'Price Change %', 'Price Difference'])
        df['Price Change %'] = df['Price Change %'].round(2)  # Round to 2 decimal places for clarity
        st.table(df)
    #     
    

elif selected_page == "HEATMAP Volume":
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    days = st.sidebar.selectbox("Select Period", ("1", "5", "7", "14","30"))
    days = int(days)
    def get_volume_data(symbol, interval, period):
        try:
            # Fetch historical data for the selected interval and period
            data = yf.download(symbol, interval=interval, period=period)
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
           
            data = data[-days:]
            # Extract the volume for each interval
            volumes = data['Volume'].values
            print(volumes)
            formatted_volumes = "{:,}".format(int(volumes))
            timestamps = data.index.strftime('%Y-%m-%d %H:%M')  # Format timestamps to show in the table
            return [symbol] + list(formatted_volumes), list(timestamps)
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None, None

    volume_data = []
    timestamp_data = []
    for symbol in stock_symbols:
        volumes, timestamps = get_volume_data(symbol, interval, period)
        if volumes is not None and timestamps is not None:
            volume_data.append(volumes)
            timestamp_data = timestamps

    if volume_data:
        # Create a DataFrame
        df = pd.DataFrame(volume_data, columns=['Symbol'] + timestamp_data[:len(volume_data[0]) - 1])
        st.table(df)
