from __future__ import division
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
# from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def cross_validate(X, y):
    kf = KFold(len(X), n_folds=10, shuffle=True)
    print X.columns
    scores = []
    conf = []
    for train, test in kf:
        X_train, X_test, y_train, y_test = X.as_matrix()[train], X.as_matrix()[test], y.as_matrix()[train], y.as_matrix()[test]
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        conf.append(confusion_matrix(y_test, clf.predict(X_test)))
        scores.append(clf.score(X_test, y_test))
    print np.average(scores)
    confusion = np.zeros((2,2))
    for i in range(len(conf)): confusion += conf[i]
    print confusion


def split_data_tt(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels)

def feature_extraction(df):
    return df.drop(['date', 'expdt', 'call', 'put', 'underlying', 'PX_EXP', 'moneyness', 'profitability', 'payoff', 'Unnamed: 0'], 1), df['moneyness']

def get_data(path='./options_data.csv'):
    df = pd.read_csv(path)
    return df

if __name__ == '__main__':
    df = get_data()
    X, y = feature_extraction(df)
    cross_validate(X, y)