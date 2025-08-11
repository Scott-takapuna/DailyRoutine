import pandas as pd

import os

import yfinance as yf

import numpy as np

from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline

import logging
from joblib import Parallel, delayed



# === Configuration ===

TICKER_FILE = 'C:\\Users\\User\\OneDrive\\Momentum\\Scripts\\Deep_Learn_Analysis\\Parameter_Library\\Params_MASTER.xlsx'

years = 7

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

logger = logging.getLogger(__name__)





def get_clean_financial_data(ticker, start_date, end_date):

    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    data.columns = data.columns.get_level_values(0)

    data = data.ffill()

    data.index = data.index.tz_localize(None)

    return data



def prepare_data(data):

    data = data.reset_index()

    data['Date_Ordinal'] = pd.to_numeric(data['Date'].map(pd.Timestamp.toordinal))

    scaler = StandardScaler()

    data['Close_Scaled'] = scaler.fit_transform(data[['Close']])

    X = data[['Date_Ordinal']].values

    y = data['Close_Scaled'].values

    return X, y, data, scaler



def train_models_rolling(X, y, n_components, degree, train_window):

    y_pred = np.full_like(y, np.nan)

    for i in range(train_window, len(X)):

        X_train = X[i-train_window:i]

        y_train = y[i-train_window:i]

        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)

        gmm.fit(X_train)

        latent_features = gmm.predict_proba(X_train)

        X_latent = np.hstack([X_train, latent_features])

        poly_reg = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

        poly_reg.fit(X_latent, y_train)

        current_latent = gmm.predict_proba(X[i].reshape(1, -1))

        current_X_latent = np.hstack([X[i].reshape(1, -1), current_latent])

        y_pred[i] = poly_reg.predict(current_X_latent)[0]

    return y_pred



def generate_signals_rolling(data, y_pred, train_window, std_multiplier):

    window = train_window

    residuals = data['Close'] - y_pred

    data['Upper_Bound'] = np.nan

    data['Lower_Bound'] = np.nan

    for i in range(window, len(data)):

        current_std = np.std(residuals.iloc[i-window:i])

        data.loc[data.index[i], 'Upper_Bound'] = y_pred[i] + std_multiplier * current_std

        data.loc[data.index[i], 'Lower_Bound'] = y_pred[i] - std_multiplier * current_std

    data['Buy_Signal'] = np.where(data['Close'] < data['Lower_Bound'], 1, 0)

    data['Sell_Signal'] = np.where(data['Close'] > data['Upper_Bound'], 1, 0)

    return data



def process_ticker_to_excel(ticker, start_date, end_date, n_components, degree, std_multiplier, train_window, rank, sector, industry, title, trades, Strat_return, Stop_loss):
    alerts = []
    try:

        logger.info(f"Processing {ticker}")

        data = get_clean_financial_data(ticker, start_date, end_date)

        X, y, data, scaler = prepare_data(data)

        y_pred_scaled = train_models_rolling(X, y, n_components, degree, train_window)

        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        data['Predicted'] = y_pred

        data = generate_signals_rolling(data, y_pred, train_window, std_multiplier)

        # === Extract recent signals (preceding day) ===

        recent_cutoff = pd.Timestamp.today() - pd.Timedelta(days=4)

        recent_signals = data[(data['Date'] >= recent_cutoff) & ((data['Buy_Signal'] == 1) | (data['Sell_Signal'] == 1))]

        for _, row in recent_signals.iterrows():

            signal_type = 'BUY' if row['Buy_Signal'] == 1 else 'SELL'

            alerts.append({

                'Ticker': ticker,

                'Title': title,

                'n_components': n_components,

                'degree': degree,

                'std_multiplier': std_multiplier,

                'train_window': train_window,

                'Signal_Type': signal_type,

                'Return': Strat_return,

                'Transactions': trades,

                'Stop Loss': Stop_loss,

                'Date': row['Date'].strftime('%Y-%m-%d'),

                'Price': row['Close'],

                'Rank' : rank,

                'Industry' : industry,

                'Sector' : sector,

            })

 
        
        return alerts

    except Exception as e:

        logger.exception(f"Error processing {ticker}: {e}")
        return alerts

        

def main():    

    tickers_df = pd.read_excel(TICKER_FILE)

    start_date = pd.Timestamp.today() - pd.DateOffset(years=years)

    end_date = pd.Timestamp.today()
    def process_row(row):
        ticker = row['Ticker']
        try:
            n_components = int(row['n_components'])
            degree = int(row['poly_degree'])
            std_multiplier = float(row['std_multiplier'])
            train_window = int(row['train_window'])
            rank = int(row['Rank'])
            sector = row['Sector']
            industry = row['Industry']
            title = row['Title']
            trades = int(row['# of trades'])
            Strat_return = float(row['Strategy Run Rate'])
            Stop_loss = float(row['Min_StopLoss%'])
            print(f"🔍 Processing {rank}. {ticker} with n_components={n_components}, degree={degree}, std_multiplier={std_multiplier}, train_window={train_window}")
            return process_ticker_to_excel(ticker, start_date, end_date, n_components, degree, std_multiplier, train_window, rank, sector, industry, title, trades, Strat_return, Stop_loss)
        except Exception as e:
            print(f"⚠️ Skipping {ticker} due to missing/invalid parameters: {e}")
            return []

    results = Parallel(n_jobs=-1)(delayed(process_row)(row) for _, row in tickers_df.iterrows())
    signal_alerts = [alert for sublist in results for alert in sublist]


          

    # === Write Signal Alert CSV with timestamp ===

    if signal_alerts:

        alerts_df = pd.DataFrame(signal_alerts)

        alerts_df.sort_values(by='Date', ascending=False, inplace=True)



        alert_filename = f'Signal_Archive/Signal_Alert_from_Market_Scanner.csv'

        alert_log = f'Signal_Archive/Signal_Alert_log.csv'

        

        alerts_df.to_csv(alert_filename, index=False)

        

        if os.path.exists(alert_log):

            # Append to the existing file without writing the header

            alerts_df.to_csv(alert_log, mode='a', header=False, index=False)

        else:

        # If the file does not exist, create it and write the header

            alerts_df.to_csv(alert_log, mode='w', header=True, index=False)



        print(f"\n🚨 Signal Alert CSV created: {alert_filename} ({len(alerts_df)} recent signals found)\n")

    else:

        print("\n✅ No new signals in the last 2 days. No Signal Alert file created.\n")

        

if __name__ == '__main__':

    main()
