import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pandas_ta as ta
# List of Indian stocks
stock_symbols = ['GRANULES.NS','TATACOMM.NS','SRF.NS','SAIL.NS','TRENT.NS','MARUTI.NS','JSWSTEEL.NS','JUBLFOOD.NS','GAIL.NS','HDFCLIFE.NS','POLYCAB.NS','ASIANPAINT.NS','LUPIN.NS','ICICIGI.NS','GLENMARK.NS','SUNTV.NS','MFSL.NS','SYNGENE.NS','OBEROIRLTY.NS','ATUL.NS','MCX.NS','ICICIPRULI.NS','NTPC.NS','TATAPOWER.NS','TORNTPHARM.NS','AUROPHARMA.NS','DLF.NS','IEX.NS','ZYDUSLIFE.NS','ALKEM.NS','HINDALCO.NS','BALKRISIND.NS','MRF.NS','NATIONALUM.NS','SBILIFE.NS','ABCAPITAL.NS','TVSMOTOR.NS','ADANIPORTS.NS','GMRINFRA.NS','SIEMENS.NS','BAJAJ-AUTO.NS','HAVELLS.NS','BHARTIARTL.NS','GODREJPROP.NS','HINDCOPPER.NS','MUTHOOTFIN.NS','SHREECEM.NS','PEL.NS','PETRONET.NS','ADANIENT.NS','ASHOKLEY.NS','HEROMOTOCO.NS','CUMMINSIND.NS','ABFRL.NS','KOTAKBANK.NS','VEDL.NS','JINDALSTEL.NS','IPCALAB.NS','MOTHERSON.NS','ONGC.NS','UPL.NS','DIXON.NS','AARTIIND.NS','BERGEPAINT.NS','LTTS.NS','PERSISTENT.NS','CIPLA.NS','COLPAL.NS','CROMPTON.NS','PIIND.NS','AMBUJACEM.NS','PIDILITIND.NS','ITC.NS','COALINDIA.NS','LALPATHLAB.NS','TATASTEEL.NS','GUJGASLTD.NS','SUNPHARMA.NS','EICHERMOT.NS','METROPOLIS.NS','LAURUSLABS.NS','ABBOTINDIA.NS','TECHM.NS','TATACHEM.NS','ABB.NS','BIOCON.NS','SBICARD.NS','HCLTECH.NS','BATAINDIA.NS','ICICIBANK.NS','LT.NS','ULTRACEMCO.NS','BPCL.NS','DEEPAKNTR.NS','GODREJCP.NS','TCS.NS','COROMANDEL.NS','SHRIRAMFIN.NS','IDEA.NS','NAUKRI.NS','BALRAMCHIN.NS','DABUR.NS','ESCORTS.NS','CHAMBLFERT.NS','NESTLEIND.NS','HINDUNILVR.NS','NAVINFLUOR.NS','BAJAJFINSV.NS','BOSCHLTD.NS','ACC.NS','CHOLAFIN.NS','DIVISLAB.NS','INDUSINDBK.NS','SBIN.NS','HDFCAMC.NS','MANAPPURAM.NS','RAMCOCEM.NS','VOLTAS.NS','RECLTD.NS','IDFC.NS','INDIGO.NS','LTF.NS','PFC.NS','UNITDSPR.NS','COFORGE.NS','GNFC.NS','HDFCBANK.NS','IDFCFIRSTB.NS','ASTRAL.NS','FEDERALBNK.NS','NMDC.NS','BAJFINANCE.NS','TITAN.NS','IRCTC.NS','DALBHARAT.NS','RELIANCE.NS','CUB.NS','LTIM.NS','TATAMOTORS.NS','AXISBANK.NS','M&M.NS','APOLLOTYRE.NS','INFY.NS','INDHOTEL.NS','PAGEIND.NS','CANFINHOME.NS','POWERGRID.NS','BEL.NS','HAL.NS','BHEL.NS','APOLLOHOSP.NS','BANDHANBNK.NS','M&MFIN.NS','LICHSGFIN.NS','BANKBARODA.NS','UBL.NS','IGL.NS','GRASIM.NS','MPHASIS.NS','BHARATFORG.NS','DRREDDY.NS','IOC.NS','TATACONSUM.NS','HINDPETRO.NS','MARICO.NS','OFSS.NS','CONCOR.NS','RBLBANK.NS','BRITANNIA.NS','AUBANK.NS','INDIACEM.NS','CANBK.NS','PVRINOX.NS','MGL.NS','EXIDEIND.NS','PNB.NS','JKCEMENT.NS','INDUSTOWER.NS','BSOFT.NS','INDIAMART.NS']
 
# Function to fetch and calculate EMAs
def get_ema_crossovers(symbol):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Get data for the last day
    data = yf.download(symbol, interval='15m', period='1mo')
    stock = yf.Ticker(symbol)
    upcoming_dividends = stock.dividends
    next_dividend_date = None
    days_left = 0
    
    if not upcoming_dividends.empty:
        # Get the most recent dividend date
        next_dividend_date = upcoming_dividends.index[-1].strftime('%Y-%m-%d')
        next_dividend_datetime = datetime.strptime(next_dividend_date, '%Y-%m-%d')
        
        # Calculate days left until the next dividend date
        today = datetime.today()
        if next_dividend_datetime > today:
            days_left = (next_dividend_datetime - today).days
        else:
            days_left = 0
    print(days_left , next_dividend_date)
    if data.empty:
        return None
 
    data['EMA_40'] = ta.ema(data['Close'], length=40)  # Use pandas_ta to calculate SMA
    data['EMA_89'] = ta.ema(data['Close'], length=89)
 
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
 
    # Ensure that the closest time is within the 5-minute intervals of the data
    # if abs(closest_time - fifteen_minutes_ago) > timedelta(minutes=15):
    #     print('3')
    #     return None
 
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
    # print(last_two['EMA_40'].iloc[-1] ,' || ', last_two['EMA_89'].iloc[-1] , " || ", percent_start ,last_two['EMA_40'].iloc[-1] <= last_two['EMA_89'].iloc[-1] , (last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1] ) <= percent_start , (last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1]) >= 0 ,last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1])
    # print(last_two['EMA_40'].iloc[-1] ,' || ', last_two['EMA_89'].iloc[-1] ,' || ', last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1] ,' || ', symbol)
    if ((last_two['EMA_40'].iloc[-1] <= last_two['EMA_89'].iloc[-1]) and (abs(last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1]) >= 0 ) and (abs(last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1] ) <= percent_start)):
        print(last_two['EMA_40'].iloc[-1] ,' || ', last_two['EMA_89'].iloc[-1] ,' || ', last_two['EMA_40'].iloc[-1] - last_two['EMA_89'].iloc[-1] ,' || ', symbol, ' || ', percent_start ,' || \n',last_two.tail(1))
        # return last_two.tail(1)
    # if (last_two['EMA_40'].iloc[-2] <= last_two['EMA_89'].iloc[-2]) and (last_two['EMA_40'].iloc[-1] > last_two['EMA_89'].iloc[-1]):
    #     return last_two.tail(1)
 
    return None
 
# Check for buy signals across all stocks
buy_signals = {}
for symbol in stock_symbols:
    signal = get_ema_crossovers(symbol)
    if signal is not None:
        buy_signals[symbol] = signal
 
# Print buy signals
for symbol, signal in buy_signals.items():
    print(f"Buy signal for {symbol}:")
    print(signal)
 
if not buy_signals:
    print("No buy signals found.")
