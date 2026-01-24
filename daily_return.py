import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore", module="joblib")

def get_data(ticker):
    
    df_orig = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True).reset_index()[['Date', 'Close', 'High', 'Low', 'Volume']]
    df_orig['Date'] = pd.to_datetime(df_orig['Date']).dt.strftime('%Y-%m-%d')
    
    return df_orig

def ma_features(df): 
    
    df = df.sort_values(by='Date', ascending=True)

    # =======================
    # Basic SMAs and Ratios
    # =======================
    sma_windows = [10, 25, 50, 100, 200]
    for sma_window in sma_windows:
        
        df[f'SMA_{sma_window}'] = df['Close'].rolling(window=sma_window).mean()

        # Current close relativet to n_day high | max 1
        df[f'Close_Rel_Max{sma_window}'] = (df['Close'] / df['High'].rolling(window=sma_window).max()).round(2)
        # Current close relativet to n_day low | min 1
        df[f'Close_Rel_Min{sma_window}'] = (df['Close'] / df['Low'].rolling(window=sma_window).min()).round(2)

    lag_periods = [10, 25, 50, 100, 150, 200]
    for sma_window in sma_windows:
        new_cols = {}
        for col in df.columns:
            if col == f'SMA_{sma_window}':
                for lag in lag_periods:
                        new_cols[f'{col}_Lag{lag}_min'] = (df[col] / df[col].rolling(window=lag).min()).round(2)
                        new_cols[f'{col}_Lag{lag}_max'] = (df[col] / df[col].rolling(window=lag).max()).round(2)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    for window in [50, 100, 200]:
        df[f'num_days_{window}'] = 0
        for i in range(1, len(df)):
            prev = df.loc[i - 1, f'num_days_{window}']
            price = df.loc[i, 'Close']
            sma = df.loc[i, f'SMA_{window}']
            if price > sma:
                df.loc[i, f'num_days_{window}'] = prev + 1 if prev >= 0 else 0
            elif price < sma:
                df.loc[i, f'num_days_{window}'] = prev - 1 if prev <= 0 else 0
            else:
                df.loc[i, f'num_days_{window}'] = 0

    # ============================
    # Relative Position Features
    # ============================
    def rows_since_max(x): return len(x) - x.argmax() - 1
    def rows_since_min(x): return len(x) - x.argmin() - 1

    for window in [10, 30, 60, 120, 240]:

        df[f'Rel_Max_{window}'] = (df['High'] / df['High'].rolling(window=window).max()).round(2)
        df[f'Rel_Min_{window}'] = (df['Low'] / df['Low'].rolling(window=window).min()).round(2)
        df[f'Max_{window}_Rows_Since'] = df['High'].rolling(window=window).apply(rows_since_max, raw=True)
        df[f'Min_{window}_Rows_Since'] = df['Low'].rolling(window=window).apply(rows_since_min, raw=True)

    for a, b in [(50, 100), (50, 200), (100, 200), (10, 25), (10, 50), (10, 100), (10, 200), (25, 50), (25, 100), (25, 200)]:    
        df[f'{a}_SMA_{b}'] = (df[f'SMA_{a}'] / df[f'SMA_{b}']).round(2)

    for window in sma_windows:

        df[f'SMA_{window}'] = (df['Close'] / df[f'SMA_{window}']).round(2)
        #df[f'EMA_{window}'] = (df['Close'] / df[f'EMA_{window}']).round(2)

    return df

def rsi(df):

    df = df.sort_values(by='Date', ascending=True)

    # RSI
    def RSI(data, period, diff):
        delta = data['Close'].diff(diff)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        RS = gain / loss
        return (100 - (100 / (1 + RS))).round(0)

    difs = [1, 3, 5]
    for dif in difs:
        df[f'RSI_14_{dif}'] = RSI(df, 14, dif)
        df[f'RSI_14_{dif}'] = RSI(df, 14, dif)
        df[f'RSI_14_{dif}'] = RSI(df, 14, dif)
        df[f'RSI_21_{dif}'] = RSI(df, 21, dif)
        df[f'RSI_21_{dif}'] = RSI(df, 21, dif)
        df[f'RSI_21_{dif}'] = RSI(df, 21, dif)

    return df

def volume(df):

    df = df.sort_values(by='Date', ascending=True)

    # ================
    # VOLUME
    # ================

    # OBV Core
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Momentum / Deviation
    windows = [5, 10]
    for w in windows:
        df[f'OBV_ROC{w}'] = df['OBV'].pct_change(periods=w).round(3)
        df[f'OBV_Z{w}'] = ((df['OBV'] - df['OBV'].rolling(w).mean()) / df['OBV'].rolling(w).std()).round(3)

    df['OBV'] = (df['OBV'] - df['OBV'].rolling(42).mean()) / df['OBV'].rolling(42).std()
    
    df['UpMask'] = df['Close'] > df['Close'].shift(1)
    df['DownMask'] = df['Close'] < df['Close'].shift(1)
    df['UpVolume'] = df['Volume'] * df['UpMask']
    df['DownVolume'] = df['Volume'] * df['DownMask']
    windows = [10, 25, 50, 100]
    #df = calculate_obv_volume_ratio(df, windows)
    z = 42
    for w in windows:
        up = df['UpVolume'].rolling(w).sum()
        down = df['DownVolume'].rolling(w).sum()

        ratio = up / down.replace(0, np.nan)
        ratio.fillna(0, inplace=True)
        ratio[up == 0] = 0  # More direct than re-rolling

        df[f'Vol_Ratio_{w}'] = ratio.round(3)
        df[f'Vol_Ratio_{w}'] = ((ratio - ratio.rolling(z).mean()) / ratio.rolling(z).std()).round(3)

    # Chaikin Money Flow (CMF)
    def CMF(data, period=20):
        mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
        mfv = mfm * data['Volume']
        cmf = mfv.rolling(window=period).sum() / data['Volume'].rolling(window=period).sum()
        return cmf.round(3)
    
    df['CMF_20'] = CMF(df, 20)
    df['CMF_10'] = CMF(df, 10)
     
    # Volume Rate of Change (VROC)
    windows = [3, 5, 10]
    for w in windows:
        df[f'VROC_{w}'] = df['Volume'].pct_change(periods=w).round(3)

    # Normalized Volume Spike
    windows = [10, 20, 40]
    for w in windows:
        df[f'Vol_Spike_{w}'] = (df['Volume'] / df['Volume'].rolling(w).median()).round(3)

    # Accumulation/Distribution Line (ADL)
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    df['ADL'] = (mfm * df['Volume']).cumsum().round(3)
    df['ADL'] = (df['ADL'] - df['ADL'].rolling(42).mean()) / df['ADL'].rolling(42).std()

    return df

def atr_adx(df):

    df = df.sort_values(by='Date', ascending=True)

    # ================
    # ATR
    # ================
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)
    windows = [7, 14, 21]
    for w in windows:
        df[f'ATR_{w}'] = tr.rolling(w).mean().round(1)
        df[f'ATR_{w}'] = (df[f'ATR_{w}'] - df[f'ATR_{w}'].rolling(42).mean()) / df[f'ATR_{w}'].rolling(42).std()

    # ================
    # ADX & DI
    # ================
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
    dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(14).mean()

    df['plus_DI'] = plus_di.round(0)
    df['minus_DI'] = minus_di.round(0)
    df['ADX'] = adx.round(0)

    return df

def volatility(df):

    df = df.sort_values(by='Date', ascending=True)

    # ================
    # Volatility
    # ================
    vol_5 = df['Close'].pct_change().rolling(window=5).std().round(3)
    vol_10 = df['Close'].pct_change().rolling(window=10).std().round(3)
    vol_25 = df['Close'].pct_change().rolling(window=25).std().round(3)

    new_cols = {
        'vol_5': vol_5,
        'vol_10': vol_10,
        'vol_25': vol_25,
        'Price_Vol_Ratio_5': (df['Close'] / vol_5).round(3),
        'Price_Vol_Ratio_10': (df['Close'] / vol_10).round(3),
        'Price_Vol_Ratio_25': (df['Close'] / vol_25).round(3)
    }

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Z-score normalize each
    for w in [5, 10, 25]:
        vol_col = f'vol_{w}'
        df[f'{vol_col}'] = (
            (df[vol_col] - df[vol_col].rolling(42).mean()) /
            df[vol_col].rolling(42).std()
        ).replace([np.inf, -np.inf], 0).fillna(0).round(3)

        pvol_col = f'Price_Vol_Ratio_{w}'
        df[f'{pvol_col}'] = (
            (df[pvol_col] - df[pvol_col].rolling(42).mean()) /
            df[pvol_col].rolling(42).std()
        ).replace([np.inf, -np.inf], 0).fillna(0).round(3)

    return df

def vix_skew(df):

    df = df.sort_values(by='Date', ascending=True)

    # ================
    # VIX External Data
    # ================
    vix_data = yf.Ticker("^VXN").history(period="max", interval="1d", auto_adjust=True)
    # Resetting the index will turn the Date index into a column
    vix_data = vix_data.reset_index()[['Date', 'Close', 'High', 'Low', 'Volume']]
    # Convert the Date column to 'YYYY-MM-DD' format (if not already)
    vix_data['Date'] = pd.to_datetime(vix_data['Date']).dt.strftime('%Y-%m-%d')
    vix_data['VIX'] = vix_data['Close']

    ['VIX_5_change', 'VIX_crossover', 'VIX_1_change']
    # Convert the Date column to 'YYYY-MM-DD' format (if not already)
    vix_data = vix_data.sort_index(ascending=True)
    vix_data['VIX_rolling_std'] = vix_data['VIX'].rolling(window=5).std().round(1)
    vix_data['VIX_short_ma'] = vix_data['VIX'].rolling(window=3).mean()
    vix_data['VIX_long_ma'] = vix_data['VIX'].rolling(window=20).mean()
    vix_data['VIX_crossover'] = np.where(vix_data['VIX_short_ma'] > vix_data['VIX_long_ma'], 1, -1)
    vix_data['VIX_5_change'] = vix_data['VIX'].pct_change(periods=5).round(3)
    vix_data['VIX_1_change'] = vix_data['VIX'].pct_change(periods=1).round(3)
    vix_data['VIX_10_change'] = vix_data['VIX'].pct_change(periods=10).round(3)

    df = pd.merge(df, vix_data[['Date', 'VIX', 'VIX_rolling_std', 'VIX_crossover', 'VIX_5_change', 'VIX_1_change', 'VIX_10_change']],
                    on='Date', how='left')
    
    vix_cols = [col for col in df.columns if col.startswith('VIX')]
    df[vix_cols] = df[vix_cols].ffill()
    
    # ================
    # SKEW External Data
    # ================
    skew_data = yf.Ticker("^SKEW").history(period="max", interval="1d", auto_adjust=True)
    # Resetting the index will turn the Date index into a column
    skew_data = skew_data.reset_index()[['Date', 'Close', 'High', 'Low', 'Volume']]
    # Convert the Date column to 'YYYY-MM-DD' format (if not already)
    skew_data['Date'] = pd.to_datetime(skew_data['Date']).dt.strftime('%Y-%m-%d')
    skew_data['skew'] = skew_data['Close']

    ['skew_5_change', 'skew_crossover', 'skew_1_change']
    # Convert the Date column to 'YYYY-MM-DD' format (if not already)
    skew_data = skew_data.sort_index(ascending=True)
    skew_data['skew_rolling_std'] = skew_data['skew'].rolling(window=5).std().round(1)
    skew_data['skew_short_ma'] = skew_data['skew'].rolling(window=3).mean()
    skew_data['skew_long_ma'] = skew_data['skew'].rolling(window=20).mean()
    skew_data['skew_crossover'] = np.where(skew_data['skew_short_ma'] > skew_data['skew_long_ma'], 1, -1)
    skew_data['skew_5_change'] = skew_data['skew'].pct_change(periods=5).round(3)
    skew_data['skew_1_change'] = skew_data['skew'].pct_change(periods=1).round(3)
    skew_data['skew_10_change'] = skew_data['skew'].pct_change(periods=10).round(3)

    df = pd.merge(df, skew_data[['Date', 'skew', 'skew_rolling_std', 'skew_crossover', 'skew_5_change', 'skew_1_change', 'skew_10_change']],
                    on='Date', how='left')
    
    skew_cols = [col for col in df.columns if col.startswith('skew')]
    df[skew_cols] = df[skew_cols].ffill()

    return df

def experimental_slope(df):

    df = df.sort_values(by='Date', ascending=True)
    
    # ================
    # Experimental Features
    # ================
    # CCI (Commodity Channel Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=14).mean()
    md = tp.rolling(window=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI_14'] = ((tp - ma) / (0.015 * md)).round(3)

    ma = tp.rolling(window=5).mean()
    md = tp.rolling(window=5).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI_5'] = ((tp - ma) / (0.015 * md)).round(3)

    # Williams %R
    highest_high = df['High'].rolling(window=14).max()
    lowest_low = df['Low'].rolling(window=14).min()
    df['Williams_%R_14'] = ((highest_high - df['Close']) / (highest_high - lowest_low) * -100).round(3)

    highest_high = df['High'].rolling(window=5).max()
    lowest_low = df['Low'].rolling(window=5).min()
    df['Williams_%R_5'] = ((highest_high - df['Close']) / (highest_high - lowest_low) * -100).round(3)

    # Z-scores
    zs = [5, 10, 25, 50]
    for z in zs:
        df[f'Zscore_{z}'] = ((df['Close'] - df['Close'].rolling(z).mean()) / df['Close'].rolling(z).std()).round(3)
        df[f'Zscore_{z}'] = df[f'Zscore_{z}'].replace([np.inf, -np.inf], np.nan)

    def generate_slope_features(df):
        slope_windows = [10, 25, 50]

        slope_targets = ['Close']

        def fast_slope_normalized(series, w):
            x = np.arange(w)
            x_mean = x.mean()
            denominator = ((x - x_mean) ** 2).sum()
            return series.rolling(w).apply(
                lambda y: ((x - x_mean) * ((y / y[0]) - (y / y[0]).mean())).sum() / denominator
                if y[0] != 0 else np.nan,
                raw=True
            )

        slope_features = {}
        for w in slope_windows:
            for var in slope_targets:
                if var in df.columns:
                    slope_features[f'{var}_slope{w}'] = fast_slope_normalized(df[var], w).round(5)

        return pd.concat([df, pd.DataFrame(slope_features, index=df.index)], axis=1)

    # Usage
    df = generate_slope_features(df)

    return df

def generate_returns(df, returns):

    def add_column_based_on_future_value(df, days):

        df = df.sort_values(by='Date', ascending=True)
        future_return = (df['Close'].shift(-days) - df['Close']) / df['Close']
        df[f'Return%_{days}'] = (future_return * 100).round(1) 
        df[f'Return_{days}'] = (future_return > 0).astype(int)
        
        past_return = (df['Close'] - df['Close'].shift(days)) / df['Close']
        df[f'Past_Return_{days}'] = (past_return > 0).astype(int)

        return df

    df_returns = df.copy()

    for r in returns:
    
        df_returns = add_column_based_on_future_value(df_returns, r)

    return df_returns

def generate_col_list(df):

    cols = [c for c in df.columns if c not in exclude]

    return cols

def macd(df):

    df = df.sort_values(by='Date', ascending=True)

    # ================
    # MACD
    # ================
    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (ema_fast - ema_slow).round(3)
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean().round(3)
    df['MACD'] = (df['MACD'] / df['Close']).round(4)
    df['Signal_Line'] = (df['Signal_Line'] / df['Close']).round(4)

    # ================
    # Bollinger Bands
    # ================
    bb_mid = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = (bb_mid + 2 * bb_std).round(3) / df['Close']
    df['BB_Lower'] = (bb_mid - 2 * bb_std).round(3) / df['Close']
    df['BB_Mid_raw'] = (bb_mid * bb_std / df['Close'])
    df['BB_Mid'] = ((df['BB_Mid_raw'] - df['BB_Mid_raw'].rolling(42).mean()) / df['BB_Mid_raw'].rolling(42).std()).round(3)

    return df

exclude = {'Date', 'Close', 'High', 'Low', 'Volume'}

def pull_daily(ticker, returns):
    df_orig = get_data(ticker) # extract pricing data
    df_returns = generate_returns(df_orig, returns) # generate target variable

    ma_df = ma_features(df_orig)
    ma_cols = generate_col_list(ma_df)
    rsi_df = rsi(df_orig)
    rsi_cols = generate_col_list(rsi_df)
    macd_df = macd(df_orig)
    macd_cols = generate_col_list(macd_df)
    volume_df = volume(df_orig)
    volume_cols = generate_col_list(volume_df)
    atr_adx_df = atr_adx(df_orig)
    atr_adx_cols = generate_col_list(atr_adx_df)
    volatility_df = volatility(df_orig)
    volatility_cols = generate_col_list(volatility_df)
    vix_skew_df = vix_skew(df_orig)
    vix_skew_cols = generate_col_list(vix_skew_df)
    experimental_slope_df = experimental_slope(df_orig)
    experimental_slope_cols = generate_col_list(experimental_slope_df)

    feature_sets = [ma_df, rsi_df, macd_df, volume_df, atr_adx_df, volatility_df, 
                    vix_skew_df, experimental_slope_df]
    feature_cols = {
    "ma": ma_cols,
    "rsi": rsi_cols,
    "macd": macd_cols,
    "volume": volume_cols,
    "atr_adx": atr_adx_cols,
    "volatility": volatility_cols,
    "vix_skew": vix_skew_cols,
    "experimental_slope": experimental_slope_cols,
    }

    # merge returns and features table into one df
    df_merged = pd.merge(ma_df, df_returns[[col for col in df_returns.columns if col.startswith('Return')] + ['Date']], on='Date') 

    # merge challengers, skipping duplicates
    for feature_set in feature_sets:
        new_cols = [c for c in feature_set.columns if c not in df_merged.columns or c == 'Date']
        df_merged = pd.merge(df_merged, feature_set[new_cols], on='Date')

    df = df_merged.sort_values('Date', ascending=False).copy()  # newest first

    return df, feature_cols