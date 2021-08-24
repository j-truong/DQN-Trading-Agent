import pandas as pd


class Strats:
    # Simple trading strategies used to compare against the DQN trading models.

    def __init__(self, data, bank_size=200, quantity_size=20):
        self.data = data
        self.buy = None
        self.sell = None
        
        self.bank = bank_size
        self.quantity = quantity_size
        
    def RSI(self, period = 14):
        # Relative Strength Index: Used to indicate whether a price is over-bought
        # or over-sold by evaluating the past timesteps.
        #
        # param     period  Length of past timesteps
        # output    rsi     Relative Strength Index

        gain, loss = self.data.close.diff(), self.data.close.diff()

        gain[gain<0] = 0
        loss[loss>0] = 0

        gain = gain.rolling(period).mean()
        loss = loss.rolling(period).mean()

        rs_gain = (period-1)*gain.shift(1) + gain
        rs_loss = (period-1)*loss.shift(1) + loss
        rsi = 100 - (100/(1 - (rs_gain/rs_loss) ))

        return rsi

    
    def MeanReversion(self, period=20):
        # Trading strategy that is based on consistent variance around
        # an underlying curve.
        #
        # param     period      Length of past timesteps
        # output    buy_price   Which prices to buy  
        # output    sell_price  Which prices to sell

        ma = self.data.close.rolling(period).mean()
        std = self.data.close.rolling(period).std()
        upper = ma + (2*std)
        lower = ma - (2*std)

        rsi = self.RSI()
        buy_price = self.data[ (self.data.close<lower) & (rsi<30) ]
        sell_price = self.data[ (self.data.close>upper) & (rsi>70) ]
        
        self.buy = buy_price
        self.sell = sell_price
        
        return buy_price, sell_price
        
        
    def ewa(self, period, smoothing=2):
        # Exponential Moving Average: calculates underlying curve for a
        # defined amount of previous timesteps.
        #
        # param     period      Length of previous timesteps
        # param     smoothing   Parametr used to smooth curve
        # output    data        Smoothened EMA of the data

        ewa = []
        for _ in range(period):
            ewa.append(-1)

        ewa.append( self.data.close.iloc[:period].mean() )

        for ts in range(period+1, len(self.data)):
            coeff = ( smoothing/(1+period) )
            ewa.append( (self.data.close.iloc[ts]*coeff) + (ewa[-1]*(1-coeff)) )

        return pd.Series(ewa, index=self.data.index)
    
    
    def MATS(self, period1=5, period2=20, period3=50):
        # Moving Average Trading Strategy: a trading strategy that uses 3 EWA of different
        # periods.
        #
        # param     period1     Period of first EWA
        # param     period2     Period of second EWA
        # param     period3     Period of third EWA
        # output    buy_price   Prices to buy
        # output    sell_price  Prices to sell

        ewa1 = self.ewa(period1)
        ewa2 = self.ewa(period2)
        ewa3 = self.ewa(period3)

        sell = (ewa1>ewa2) & (ewa1.shift(1)<ewa2.shift(1)) & (ewa1>ewa3) & (ewa2>ewa3) & (ewa3 != -1)
        buy = (ewa1<ewa2) & (ewa1.shift(1)>ewa2.shift(1)) & (ewa1<ewa3) & (ewa1<ewa3) & (ewa3 != -1)

        buy_price = self.data[buy]
        sell_price = self.data[sell]
        
        self.buy = buy_price
        self.sell = sell_price
        
        return buy_price, sell_price
    
    
    def run(self):
        # Runs the specifified trading strategy in a simulated environment and returns its history.
        #
        # output    history     History of trading strategy under simulation.

        portfolio = self.bank
        bank = self.bank
        inventory = []
        
        buy = self.buy
        sell = self.sell
        buy['action'] = 'buy'
        sell['action'] = 'sell'
        trades = pd.concat([buy,sell], axis=0).sort_index()
        
        history = pd.DataFrame(columns=['ts','price','trade','action','portfolio'])
        
        for i, trade in trades.iterrows():
            executed = False
            gain = 0
            price = trade.close*self.quantity
            
            if (trade.action == 'buy') and (price < bank):
                executed = True
                inventory.append(price)
                bank -= price
            elif (trade.action == 'sell') and inventory:
                executed = True
                prev_price = inventory.pop(0)
                portfolio += price - prev_price
                bank += price
                
            if executed:
                history = history.append({'ts':i, 'price':trade.close, 'trade':price,
                                          'action':trade.action, 'portfolio':portfolio}, 
                                         ignore_index=True)
            
        return history