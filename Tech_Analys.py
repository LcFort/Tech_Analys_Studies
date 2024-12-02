# Inspired by CHIO, Pat, 2022
# https://arxiv.org/pdf/2206.12282

# MACD and other techinal analysis indicators applied in python

import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as go
from time import time
import matplotlib.pyplot as plt


def temp(func): # Calculates the time spent running the code
    def wrapper():
        t1 = time()
        func()
        t2 = time() - t1
        print(fr'{t2:.4f} seg')
    return wrapper


class Analys_Tec():
    def __init__(self, ticker) -> None:
        self.ticker = ticker
        self.download_db()
    
    def download_db(self) -> np.array:
        base: pd.DataFrame = yf.download(self.ticker, start='2023-01-01').ffill()

        self.db: pd.DataFrame = base['Adj Close']
        self.db_return: pd.DataFrame = self.db.pct_change().fillna(0)
        
        self.prices: np.array = self.db.values
        self.dates: np.array = (self.db.index)
        self.returns: np.array = self.db_return.values
        self.volume: np.array = base['Volume'].values
        self.high: np.array = base['High'].values
        self.low: np.array = base['Low'].values
        self.close: np.array = base['Close'].values
        

    def Vol(self, returns: np.array) -> np.array:
        return np.array(returns.std())
    
    def SMA(self, window: int, returns: np.array) -> np.array:
        rolling_windows = np.lib.stride_tricks.sliding_window_view(returns, window)
        moving_average = np.mean(rolling_windows, axis = 1)
        
        agreggated = np.zeros(len(returns))
        agreggated[window-1:] = moving_average
        
        return agreggated

    def EMA(self, window: int, returns: np.array) -> np.array:
        Weight: float = 2/(window+1) # (CHIO, Pat) 2022
    
        EMA_return: np.array = np.zeros(len(returns))
        EMA_return[window-1] = returns.mean()
        
        for i in range(window, len(returns)):
            EMA_return[i] = (returns[i] * Weight) + (EMA_return[i-1] * (1 - Weight))
        
        return EMA_return

    def RS(self, window: int, price: np.array) -> np.array: # Used only in RSI, could be optimized
        Base_lenght: int = len(price)
        
        Avg_Gains: np.array = np.zeros(Base_lenght)
        Avg_Losses: np.array = np.zeros(Base_lenght)
        RS: np.array = np.zeros(Base_lenght)
        
        Price_diffs: np.array = np.insert(np.diff(price), 0, 0)
        
        # First index Avg gain and loss
        Gain_sum: float = 0
        Loss_sum: float = 0
        
        for _ in range(0, window - 1):
            if Price_diffs[_] > 0:
                Gain_sum +- Price_diffs[_]
            elif Price_diffs[_] < 0:
                Gain_sum +- abs(Price_diffs[_])
                
        Avg_Gains[window - 1] = Gain_sum
        Avg_Losses[window - 1] = Loss_sum
        
        if Gain_sum != 0 or Loss_sum != 0:
            RS[window - 1] = Gain_sum / Loss_sum
        
        for i in range(window, Base_lenght):
            Gain: float = 0
            Loss: float = 0
            
            if Price_diffs[i] > 0:
                Gain = Price_diffs[i]
            elif Price_diffs[i] < 0:
                Loss = abs(Price_diffs[i])
                
            Avg_Gains[i] = (Avg_Gains[i - 1] * (window - 1) + Gain)/window
            Avg_Losses[i] = (Avg_Losses[i - 1] * (window - 1) + Loss)/window
            
            if Avg_Gains[i] != 0 or Avg_Losses[i] != 0:
                RS[i] = Avg_Gains[i] / Avg_Losses[i]
            
        return RS

    def MACD(self) -> pd.DataFrame:
        EMA_short: np.array = self.EMA(12, self.returns)
        EMA_long: np.array = self.EMA(26, self.returns)
        
        MACD_line: np.array = EMA_short - EMA_long
        Signal: np.array = self.EMA(9, MACD_line)
        Hist: np.array = MACD_line - Signal
        
        return pd.DataFrame(data = {'EMA_short': EMA_short, 'EMA_long': EMA_long, 'MACD_line': MACD_line, 'Signal': Signal, 'Hist': Hist},
                            index = self.dates)
        
    def Bollinger_Bands(self) -> pd.DataFrame:
        window = 14
        Std: np.array = self.Vol(self.returns)
        Middle_band: np.array = self.SMA(window, self.returns)        
        Base_length: int = len(Middle_band)
        
        Upper_band: np.array = np.zeros(Base_length)
        Lower_band: np.array = np.zeros(Base_length)
        
        Upper_band[window:-1] = Middle_band[window:-1] + Std
        Lower_band[window:-1] = Middle_band[window:-1] - Std
        
        BBW: np.array = (Upper_band - Lower_band) / Middle_band
        
        Short_SMA_BBW: np.array = self.SMA(10, BBW)
        Long_SMA_BBW: np.array = self.SMA(50, BBW)
        
        return pd.DataFrame(data = {'Middle_band': Middle_band, 'Upper_band': Upper_band, 'Lower_band': Lower_band,
                                    'BBW': BBW, 'Short_SMA_BBW': Short_SMA_BBW, 'Long_SMA_BBW': Long_SMA_BBW},
                            index = self.dates)
        
    def RSI(self) -> pd.DataFrame:
        window: int = 14
        RS: np.array = self.RS(window, self.prices)
        Base_lenght: int = len(RS)
        
        RSI_line = np.empty(Base_lenght)
        RSI_line[:] = np.nan
        
        for i in range (window - 1, Base_lenght):
            RSI_line[i] = 100 - (100/(1+RS[i]))
            
        return pd.DataFrame({'RSI_line': RSI_line}, index = self.dates)

    def MFI(self) -> pd.DataFrame:
        window: int = 14
        typical_price: np.array = (self.high + self.low + self.close)/3
        money_flow: np.array = typical_price * self.volume
        Base_lenght: int = len(money_flow)
        
        base_ratio: np.array = np.zeros(Base_lenght)
        
        for i in range(1, Base_lenght - 1):
            base_ratio[i] = (money_flow[i] / money_flow[i - 1]) - 1
            
        money_flow_index: np.array = np.empty(Base_lenght)
        money_flow_index[:] = np.nan
        
        for i in range(window - 1, Base_lenght):
            ranged_data: np.array = base_ratio[(i - (window - 1)):i]
            
            money_flow_positive: int = np.sum(ranged_data > 0)
            money_flow_negative: int = np.sum(ranged_data < 0)
            
            money_flow_ratio: float = money_flow_positive / money_flow_negative
            money_flow_index[i] = 100 - 100 / (1 + (money_flow_ratio))
            
        return pd.DataFrame({'Money_Flow_Index': money_flow_index}, index = self.dates)

    def SAR(self) -> pd.DataFrame:
        alpha: float = 0.02

############################################################################################################################################################
################### Apply Trade Signals ####################################################################################################################
############################################################################################################################################################

class Apply_Signal(Analys_Tec):
    def __init__(self, ticker, wanted = {'MACD': False, 'Bollinger': False, 'RSI': False}) -> None:
        super().__init__(ticker) # Triggers the main function
        
        if wanted['MACD']:  
            df: pd.DataFrame = self.MACD() 
            self.MACD_line: np.array = df['MACD_line'].values
            self.Signal: np.array = df['Signal'].values
            self.Hist: np.array = df['Hist'].values
            self.dates: np.array = np.array(df.index)
            self.MACD_Signal = self.MACD_Analysis(self) # Creates an outer reference to call the function later
            
        if wanted['Bollinger']:
            df: pd.DataFrame = self.Bollinger_Bands()
            self.Short_SMA_BBW: np.array = df['Short_SMA_BBW'].values
            self.Long_SMA_BBW: np.array = df['Long_SMA_BBW'].values
            self.dates: np.array = np.array(df.index)
            self.Bollinger_Signal = self.Bollinger_Analysis(self)
            
        if wanted['RSI']:
            df: pd.DataFrame = self.RSI()
            self.RSI_line = df['RSI_line'].values
            self.dates: np.array = np.array(df.index)
            self.RSI_Signal = self.RSI_Analysis(self)
            
        if wanted['MFI']:
            df: pd.DataFrame = self.MFI()
            self.MFI_line: np.array = df['Money_Flow_Index'].values
            self.dates: np.array = np.array(df.index)
            self.MFI_Signal = self.MFI_Analysis(self)
            
        if wanted['SAR']:
            df: pd.DataFrame = self.SAR()
            
            self.SAR_Signal = self.SAR_Analysis(self)
        
    class MACD_Analysis:
        def __init__(self, reference) -> None:
            self.MACD_line: np.array = reference.MACD_line
            self.Signal: np.array = reference.Signal
            self.Hist: np.array = reference.Hist
            self.dates: np.array = reference.dates
        
        def Signal_crossover(self) -> pd.DataFrame: 
            Order: np.array = np.zeros(len(self.MACD_line))
            for i in range(1, len(Order)):
                if self.MACD_line[i-1] < self.Signal[i-1] and self.MACD_line[i] > self.Signal[i]:
                    Order[i] = 1
                    
                elif self.MACD_line[i-1] > self.Signal[i-1] and self.MACD_line[i] < self.Signal[i]:
                    Order[i] = -1
                    
            return pd.DataFrame({'Signal_crossover': Order}, self.dates)
        
        def Zero_crossover(self) -> pd.DataFrame:
            Order: np.array = np.zeros(len(self.MACD_line))
            
            for i in range(1, len(Order)):
                if self.MACD_line[i-1] < 0 and self.MACD_line[i] > 0:
                    Order[i] = 1
                    
                elif self.MACD_line[i-1] > 0 and self.MACD_line[i] < 0:
                    Order[i] = -1
                    
            return pd.DataFrame({'Zero_crossover': Order}, self.dates)
        
        def Histogram(self) -> pd.DataFrame:
            Order: np.array = np.zeros(len(self.MACD_line))
            
            for i in range(1 + 3, len(Order)):
                Hist_base: np.array = self.Hist[i-2:i]
                if min(Hist_base) == Hist_base[1] and all(Hist_base) < 0:
                    Order[i] = 1
                    
                elif max(Hist_base) == Hist_base[1] and all(Hist_base) > 0:
                    Order[i] = -1
                    
            return pd.DataFrame({'Histogram': Order}, self.dates)
        
        def Signal_cross_above_zero(self) -> pd.DataFrame:
            Order: np.array = np.zeros(len(self.MACD_line))
            
            for i in range(1, len(Order)):
                if self.MACD_line[i-1] < self.Signal[i-1] and all([self.MACD_line[i], self.Signal[i]]) > 0:
                    Order[i] = 1
                    
                elif self.MACD_line[i-1] > self.Signal[i-1] and all([self.MACD_line[i], self.Signal[i]]) < 0:
                    Order[i] = -1
                    
            return pd.DataFrame({'Signal_cross_above_zero': Order}, self.dates)
        
    class Bollinger_Analysis:
        def __init__(self, reference) -> None:
            self.Short_SMA_BBW: np.array = reference.Short_SMA_BBW
            self.Long_SMA_BBW: np.array = reference.Long_SMA_BBW
            self.dates: np.array = reference.dates
        
        def BBW_SMA_crossover(self) -> pd.DataFrame:
            Order: np.array = np.zeros(len(self.Short_SMA_BBW))
            for i in range(1, len(Order)):
                if self.Short_SMA_BBW[i] > self.Long_SMA_BBW[i]:
                    Order[i] = 1
                    
                elif self.Short_SMA_BBW[i] < self.Long_SMA_BBW[i]:
                    Order[i] = -1
            
            return pd.DataFrame({'BBW_SMA_crossover': Order}, self.dates)
        
    class RSI_Analysis:
        def __init__(self, reference) -> None:
            self.RSI_line: np.array = reference.RSI_line
            self.dates: np.array = reference.dates
        
        def RSI_Bounce_Over(self) -> pd.DataFrame:
            Order: np.array = np.zeros(len(self.RSI_line))
            for i in range(1, len(Order)):
                if self.RSI_line[i] <= 30:
                    Order[i] = 1
                    
                elif self.RSI_line[i] >= 70:
                    Order[i] = -1
                    
            return pd.DataFrame({'RSI_bounce_over': Order}, self.dates)
        
    class MFI_Analysis:
        def __init__(self, reference) -> None:
            self.MFI_line: np.array = reference.MFI_line
            self.dates: np.array = reference.dates
            
        def MFI_Bounce_Over(self) -> pd.DataFrame:
            Order: np.array = np.zeros(len(self.MFI_line))
            for i in range(1, len(Order)):
                if self.MFI_line[i] <= 25:
                    Order[i] = 1
                
                elif self.MFI_line[i] >= 75:
                    Order[i] = -1
                    
            return pd.DataFrame({'MFI_Bounce_Over': Order}, self.dates)
        
    class SAR_Analysis:
        def __init__(self, reference) -> None:
            pass

###
###

if __name__ == '__main__':
       
    stock = 'SPY' # pass one ticker at a time

    wanted = {'MACD': False, 'Bollinger': False, 'RSI': False, 'MFI': False, 'SAR': True}
    
    apply_signal = Apply_Signal(stock, wanted)
    
    if wanted['MACD']:
        macd_signals = apply_signal.MACD_Signal
        Sg = macd_signals.Signal_crossover()
        Zr = macd_signals.Zero_crossover()
        Hs = macd_signals.Histogram()
        Sg0 = macd_signals.Signal_cross_above_zero()

        df_complete: pd.DataFrame = Sg.join(Zr).join(Hs).join(Sg0)
        print(df_complete)
        
    if wanted['Bollinger']:
        Bollinger_signals = apply_signal.Bollinger_Signal
        SMA_BBW_c = Bollinger_signals.BBW_SMA_crossover()
        
        print(SMA_BBW_c)
        
    if wanted['RSI']:
        RSI_signals = apply_signal.RSI_Signal
        RSI_bounce_over = RSI_signals.RSI_Bounce_Over()
        
        print(RSI_bounce_over)
        
    if wanted['MFI']:
        MFI_signals = apply_signal.MFI_Signal
        MFI_bounce_over = MFI_signals.MFI_Bounce_Over()
        
        print(MFI_bounce_over)
        
    if wanted['SAR']:
        SAR_signals = apply_signal.SAR_Signal
        
    
    
    
    # fig = go.bar(Sg.join(Zr), barmode = 'group')
    # fig.show()

    #print(Sg.loc[Sg.values == Zr.values].index)