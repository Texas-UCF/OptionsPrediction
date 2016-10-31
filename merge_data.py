import pandas as pd
from OptionsBase import OptionsDataFrame
from yahoo_finance import Share
import numpy as np 

def label_df(joint_df):
    joint_df['payoff'] = joint_df['PX_EXP'] - joint_df['PX_LAST']
    joint_df['moneyness'] = joint_df['payoff'] > 0
    joint_df['profitability'] = joint_df['payoff'] > joint_df['call_close']
    return joint_df


def options_scrape():
    odf_interface = OptionsDataFrame.OptionsDataFrame()
    return odf_interface.fetch_data()


def underlying_scrape(tickers):
    full_df = pd.DataFrame()
    for ticker in tickers:
        print ticker
        share = Share(ticker)
        ts = share.get_historical('2016-01-01', '2016-11-01')
        full_df = full_df.append(pd.DataFrame(ts))
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df.index = range(len(full_df))
    return full_df[['Symbol', 'Date', 'Close']]


def join_underlying_options(options_df, underlying_df):
    joint = options_df.merge(underlying_df, left_on=['expdt', 'underlying'], right_on=['Date', 'Symbol'])
    joint = joint.drop(['Date', 'Symbol'], 1)
    joint = joint.rename(columns={'Close' : 'PX_EXP'})
    joint['PX_EXP'] = pd.to_numeric(joint['PX_EXP'], errors='coerce')
    return joint


def join_fundamental():
    pass


def fundamental_to_dataframe(matrix_df, label):
    tuple_list = []
    for row_index, row in matrix_df.iterrows():
        for col_name in matrix_df.columns:
            tuple_list += (row_index, col_name, row[col_name])
    return pd.DataFrame(tuple_list, columns=['date', 'ticker', label])


def drop_na(df):
    df = df.applymap(lambda x: np.nan if x == '<NA>' else x)
    df = df.dropna()
    return df


def merge_data():
    options_df = options_scrape()
    tickers = list(set(options_df['underlying']))
    underlying_df = underlying_scrape(tickers)
    joint = join_underlying_options(options_df, underlying_df)
    labeled_df = label_df(joint)
    labeled_df = drop_na(labeled_df)
    labeled_df.to_csv('./options_data.csv')
    return labeled_df


if __name__ == '__main__':
    merge_data()
