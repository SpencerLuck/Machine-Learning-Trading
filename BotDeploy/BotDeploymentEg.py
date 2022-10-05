import ftx
import pandas as pd
import datetime as dt
from datetime import *
from collections import deque
import numpy as np
import time
import tensorflow as tf
import tensorflow.keras as keras
import ccxt
import pandas_ta as ta

# Account info
api_key = ''
api_secret = ''
SignIn = ftx.FtxClient(api_key=api_key, api_secret=api_secret)
ftx = ccxt.ftx(
{
        'apiKey': api_key,
        'secret': api_secret
}
)


# Hyper parameters - inputs that will change the Trading Bot
# static_timestamp = None
TP_ATR = 3.0
SL_ATR = 1.5
PAIR_TO_PREDICT = 'ETH'
Market = 'ETH-PERP'
Side_Buy = "buy"
Side_Sell = "sell"
ratios = ['ETH']


# stats dictionary
stats = {'Close': {'mean': 0.00013499451463623395, 'std': 0.010222616545887233},
         'High': {'mean': 0.00012189721875011916, 'std': 0.009010927715056327},
         'Low': {'mean': 0.00014861188842568972, 'std': 0.011457198801300696},
         'Open': {'mean': 0.0001339876181819151, 'std': 0.010219478504170286},
         'Volume': {'mean': 48690641.562491134, 'std': 78657724.08329752}}

def SIGNAL1():
    return raw_df.signal1

def ATR():
    return raw_df.atr

def Neural_Network(data, path, stats_dict, seq_len):
    # preprocess
    main_df = data.copy()
    for col in main_df.columns:  # go through all of the columns
        col_end = col.rsplit('_', -1)[-1]  # get the last word of the string
        if col == "Volume" or col_end == "volume":
            main_df[col] = (main_df[col].values - stats_dict[f'{col}']['mean']) / (stats_dict[f'{col}']['std'])

        elif col_end == "value":
            main_df[col] = (main_df[col].values - stats[f'{col}']['mean']) / (stats[f'{col}']['std'])

        elif col == "Open" or col == "High" or col == "Low" or col == "Close":
            main_df[col] = (main_df[col].values - stats_dict[f'{col}']['mean']) / (stats_dict[f'{col}']['std'])

    main_df.dropna(inplace=True)  # cleanup again

    main_df.dropna(inplace=True)
    sequential_data = []
    prev_days = deque(
        maxlen=seq_len)

    for i in main_df.values:
        prev_days.append([n for n in i[:]])
        if len(prev_days) == seq_len:
            sequential_data.append(np.array(prev_days))

    X = []
    for seq in sequential_data:
        X.append(seq)

    final_data = np.array(X)

    model = tf.keras.models.load_model(path)
    predictions = model.predict(final_data)

    pred_list = []
    for pred in predictions:

        # Buy
        if (pred[2] > pred[0]) and (pred[2] > pred[1]):
            pred_list.append(2)
        # Sell
        elif (pred[0] > pred[2]) and (pred[0] > pred[1]):
            pred_list.append(0)
        # No trade
        else:
            pred_list.append(1)

    return pred_list

# model path to run locally
model_path = '/model.model'

# running the script every hour, this is for local running
# to launch this script with a cronjob the while true and time controls will need to be removed
while True:
    sleep_trade = 60 - datetime.now().minute % 60

    if sleep_trade == 60:

        # Getting Price Data
        resolution = 60 * 60
        today = dt.datetime.today()
        DayBefore = today - timedelta(days=6)
        start = DayBefore.timestamp()
        end = today.timestamp() - 60 * 60
        num_hours = 150

        # Creating the dataframe
        raw_df = pd.DataFrame()
        data = SignIn.get_historical_data(f'{Market}', resolution, num_hours, start, end)
        df = pd.DataFrame(data)
        # print(df)
        df['startTime'] = pd.to_datetime(df['startTime'])
        df.set_index('startTime', inplace=True)

        day = 60 * 60 * 24
        week = 60 * 60 * 24 * 7

        df['Day_sin'] = np.sin(df['time'] * (2 * np.pi / day))
        df['Day_cos'] = np.cos(df['time'] * (2 * np.pi / day))
        df['Week_sin'] = np.sin(df['time'] * (2 * np.pi / week))
        df['Week_cos'] = np.cos(df['time'] * (2 * np.pi / week))

        df.drop(columns=['time'], inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
                  inplace=True)
        # model_df
        model_df = df.copy()

        # model features
        # apply any features to the model_df that are specific to the neural network

        pred1 = Neural_Network(model_df, model_path, stats, 20)
        min_len = len(pred1)

        df = df.iloc[(len(df) - min_len):]
        df['signal1'] = pred1

        df['atr'] = ta.atr(high=df['High'], low=df['Low'], close=df['Close'], append=True)
        df.dropna(inplace=True)

        if len(raw_df) == 0:
            raw_df = df
        else:
            raw_df = raw_df.join(df)

        raw_df.fillna(method="ffill", inplace=True)
        raw_df.dropna(inplace=True)

        # Get the latest market info
        MarketInfo = SignIn.get_market(Market)
        # Get the latest market price to place a buy order
        MarketPrice = MarketInfo['last']
        # Latest Bid and Ask Price
        Bid_Ask = SignIn.get_orderbook(market=Market)

        # Closing Trades
        Existing_Order = pd.DataFrame(SignIn.get_positions())
        Com_Order = pd.DataFrame(SignIn.get_conditional_orders())
        if Market in Existing_Order['future'].unique():
            Existing_Position_Size = float(Existing_Order.loc[Existing_Order['future'] == f'{Market}']['size'])
            if Existing_Position_Size > 0:
                print('Checking to close open trade')
                # determines whether it is a buy or sell pos
                Existing_Order = Existing_Order.loc[Existing_Order['future'] == Market]
                Existing_Order['side'] = Existing_Order['side'].astype("string")
                Old_Order_Side = Existing_Order["side"].item()

                # determines the old pos size that was held in order to close the entire position
                Existing_Order["size"] = Existing_Order["size"].astype("string")
                Old_Order_Size = Existing_Order["size"].item()

                if Old_Order_Side == "buy" and (SIGNAL1()[-1] == 0):
                    print("The Existing Order is a BUY and needs to be closed with a SELL")
                    Close = ftx.create_order(symbol=Market, type='market', side=Side_Sell, amount=Old_Order_Size,
                                             params={'reduceOnly': True})
                    CancelOrders = SignIn.cancel_orders(market_name=Market, conditional_orders=True)
                elif Old_Order_Side == "sell" and (SIGNAL1()[-1] == 2):
                    print("The Existing Order is a SELL and needs to be close with a BUY")
                    Close = ftx.create_order(symbol=Market, type='market', side=Side_Buy, amount=Old_Order_Size,
                                             params={'reduceOnly': True})
                    CancelOrders = SignIn.cancel_orders(market_name=Market, conditional_orders=True)
                else:
                    print('Trade open, no exit signal')
            else:
                if len(Com_Order) == 0:
                    print("No orders open and no conditional orders")
                elif Market in Com_Order['future'].unique():
                    print('No orders open and cancelling conditional orders')
                    CancelOrders = SignIn.cancel_orders(market_name=Market, conditional_orders=True)

        # Ask prices will be used for the trigger sell
        buy_tp = round((Bid_Ask['asks'][0][0]) + (TP_ATR * ATR()[-1]), 1)
        buy_sl = round((Bid_Ask['asks'][0][0]) - (SL_ATR * ATR()[-1]), 1)
        # Bid prices will be used for the trigger buy
        sell_tp = round((Bid_Ask['bids'][0][0]) - (TP_ATR * ATR()[-1]), 1)
        sell_sl = round((Bid_Ask['bids'][0][0]) + (SL_ATR * ATR()[-1]), 1)

        # Calculate Pos Size for new trades
        Acc = SignIn.get_account_info()
        Acc_Bal = Acc['totalAccountValue']
        Pos_size_buy = round(0.01 * float(Acc_Bal) / (Bid_Ask['bids'][0][0] - buy_sl), 3)
        Pos_size_sell = round(0.01 * float(Acc_Bal) / (sell_sl - Bid_Ask['asks'][0][0]), 3)


        min_size = 0.001  # ETH specific
        if Pos_size_buy < min_size:
            Pos_size_buy = min_size
        if Pos_size_sell < min_size:
            Pos_size_sell = min_size

        # checking if there are positions open to avoid doubling pos sizes
        OpenTrades = 0 # set open trades to 0
        if Market in Existing_Order['future'].unique():
            Existing_Position_Size = float(Existing_Order.loc[Existing_Order['future'] == f'{Market}']['size'])
            if Existing_Position_Size > 0:
                OpenTrades = 1
                print("Trade Already Open, OpenTrades=1")
        if OpenTrades == 0:
            if SIGNAL1()[-1] == 2:
                if SIGNAL1()[-2] != 2:
                    print('Buy signal')
                    BuyOrder = ftx.create_order(symbol=Market, type='market', side='buy', amount=Pos_size_buy)

                    StopLossBuy = ftx.create_order(symbol=Market, side='sell', type='stop', amount=Pos_size_buy,
                                                   params={'triggerPrice': buy_sl, 'reduceOnly': True})

                    TakeProfitBuy = ftx.create_order(symbol=Market, side='sell', type='takeProfit', amount=Pos_size_buy,
                                                      params={'triggerPrice': buy_tp, 'reduceOnly': True})

                else:
                    print('No buy signal')
            elif SIGNAL1()[-1] == 0:
                if SIGNAL1()[-2] != 0:
                    print('Sell signal')
                    SellOrder = ftx.create_order(symbol=Market, type='market', side='sell', amount=Pos_size_sell)

                    StopLossSell = ftx.create_order(symbol=Market, side='buy', type='stop', amount=Pos_size_sell,
                                                    params={'triggerPrice': sell_sl, 'reduceOnly': True})

                    TakeProfitSell = ftx.create_order(symbol=Market, side='buy', type='takeProfit', amount=Pos_size_sell,
                                                       params={'triggerPrice': sell_tp, 'reduceOnly': True})
                else:
                    print('No sell signal')
            else:
                print('Flat signal: No trades')

        time.sleep(sleep_trade * 60)
    else:
        time.sleep(sleep_trade * 60)
