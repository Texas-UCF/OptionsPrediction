from __future__ import division
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
# from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def cross_validate(X, y, pca_reduce=True):
    if pca_reduce == True:
        X = pd.DataFrame(dimensionality_reduction(X, y))
    kf = KFold(len(X), n_folds=10, shuffle=True)
    accuracies = []
    conf = []
    precisions = []
    recalls = []
    for train, test in kf:
        X_train, X_test, y_train, y_test = X.as_matrix()[train], X.as_matrix()[test], y.as_matrix()[train], y.as_matrix()[test]
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        conf.append(confusion_matrix(y_test, predictions))
        recalls.append(recall_score(y_test, predictions))
        precisions.append(precision_score(y_test, predictions))
        accuracies.append(clf.score(X_test, y_test))
    print 'Accuracy: ' + str(np.average(accuracies))
    print 'Precision: ' + str(np.average(precisions))
    print 'Recall: ' + str(np.average(recalls))
    confusion = np.zeros((2,2))
    for i in range(len(conf)):
        confusion += conf[i]
    print confusion

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_test[:], predictions[:])
    roc_auc[0] = auc(fpr[0], tpr[0])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    average_precision = dict()
    precision = dict()
    recall = dict()
    precision[0], recall[0], _ = precision_recall_curve(y_test[:], predictions[:])
    average_precision[0] = average_precision_score(y_test[:], predictions[:])
    plt.clf()
    plt.plot(recall[0], precision[0], lw=lw, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()


def dimensionality_reduction(X, y, components=5):
    pca = PCA(n_components=components)
    X = pca.fit_transform(X,y)
    print pca.explained_variance_ratio_
    return X

def split_data_tt(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels)

def feature_extraction(df):
    labels = df['moneyness']
    categorical_columns = ['underlying', 'INDUSTRY_SECTOR', 'INDUSTRY_GROUP', 'INDUSTRY_SUBGROUP']
    exclude = ['date', 'expdt', 'call', 'put', 'PX_EXP', 'moneyness', 'profitability', 'payoff', 'Price'] + categorical_columns
    feature_df = df.drop(exclude, 1)
    feature_df = feature_df.drop(['Unnamed: 0'], axis=1) if 'Unnamed: 0' in feature_df.columns else feature_df
    ohe = pd.get_dummies(df[categorical_columns], columns=categorical_columns)
    feature_df = pd.concat([feature_df, ohe], axis=1)
    return feature_df, labels

def get_data(path='./options_data.csv'):
    df = pd.read_csv(path)
    return df

if __name__ == '__main__':
    df = get_data('./options_fundamental_data.csv')
    X, y = feature_extraction(df)
    cross_validate(X, y, False)