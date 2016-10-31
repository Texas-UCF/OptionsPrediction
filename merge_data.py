import pandas as pd
from OptionsBase import OptionsDataFrame
from yahoo_finance import Share
import numpy as np
import xlrd
import csv
import os

def label_df(joint_df):
    joint_df['payoff'] = joint_df['PX_EXP'] - joint_df['strike']
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


def fundamental_xls_to_csv():
    book = xlrd.open_workbook('earnings.xlsx')
    for sheet in book.sheets():
        fp = open('fundamental/' + sheet.name + '.csv', 'wb')
        wr = csv.writer(fp)
        for rownum in xrange(sheet.nrows):
            wr.writerow([unicode(val).encode('utf8') for val in sheet.row_values(rownum)])


def join_fundamental(full_df):
    for filename in os.listdir('./fundamental'):
        col_name = filename[:-4]
        print col_name
        df = pd.read_csv('./fundamental/' + filename, index_col=0)
        if filename in ['Industry.csv']:
            fundamental_df = industry_to_dataframe(df)
            cols = fundamental_df.columns[:-1]
            full_df[cols] = full_df.merge(fundamental_df, left_on=['underlying'], right_on=['ticker'])[cols]
            continue

        if df.index.name == 'Date':
            df = drop_na(df, 'Date')
            df.index = df.index.map(lambda x: xldate_to_datetime(x))
        for col in df.columns:
            df[col] = df[col].interpolate()
        df = interpolate_df(df)
        fundamental_df = fundamental_to_dataframe(df, col_name)

        full_df[col_name] = full_df.merge(fundamental_df, left_on=['date', 'underlying'], right_on=['date', 'ticker'])[col_name]
    full_df = drop_na(full_df)
    return full_df


def xldate_to_datetime(xldate):
    year, month, day, _, _, _ = xlrd.xldate_as_tuple(xldate, 0)
    date_str = str(year) + str(month) + str(day)
    return pd.to_datetime(date_str, format='%Y%m%d')


def drop_na(df, index=False):
    df = df.applymap(lambda x: np.nan if x in ('<NA>', '#N/A', float('NaN')) else x)
    if index:
        df = df[np.isfinite(df.index)]
    else:
        df = df.dropna()
    return df


def interpolate_df(df):
    date_range = pd.bdate_range(min(df.index), max(df.index))
    date_df = pd.DataFrame(index=date_range)
    new_df = df.join(date_df, how='right')
    for col in new_df.columns:
        new_df[col] = new_df[col].interpolate()
    return new_df


def fundamental_to_dataframe(matrix_df, label, col1='date', col2='ticker'):
    tuple_list = []
    for row_index, row in matrix_df.iterrows():
        for col_name in matrix_df.columns:
            tuple_list += [(row_index, col_name, row[col_name])]
    return pd.DataFrame.from_records(tuple_list, columns=[col1, col2, label])


def industry_to_dataframe(matrix_df):
    df = matrix_df.T
    df['ticker'] = matrix_df.columns
    df['ticker'] = df['ticker'].apply(lambda x: x.split()[0])
    return df


def merge_data():
    options_df = options_scrape()
    tickers = list(set(options_df['underlying']))
    underlying_df = underlying_scrape(tickers)
    joint = join_underlying_options(options_df, underlying_df)
    labeled_df = label_df(joint)
    labeled_df = drop_na(labeled_df)
    labeled_df.to_csv('./options_data.csv', index_label=False)
    return labeled_df


def load_labeled_data(path='./options_data.csv'):
    df = pd.read_csv(path)
    df['expdt'] = pd.to_datetime(df['expdt'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(['Unnamed: 0'], axis=1) if 'Unnamed: 0' in df.columns else df
    return df


if __name__ == '__main__':
    df = load_labeled_data()
    full_df = join_fundamental(df)
    full_df.to_csv('./options_fundamental_data.csv')
