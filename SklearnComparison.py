
import numpy

from numpy import genfromtxt
from sklearn import ensemble
from sklearn import feature_selection
from sklearn import metrics
from sklearn import model_selection

from sklearn.impute import SimpleImputer




def convert1(x: str):
    if (x==b'NORMAL'):
        return 0
    if (x==b'RECOVERING'):
        return 1
    return 2

def convert2(x: str):
    if (x==b'NA'):
        return numpy.nan
    return float(x)

def getConvertFuncs(n):
    d = {}
    for i in range(0, n):
        d[i] = convert2
    return d

def classify(data, labels):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels)
    classifier = ensemble.RandomForestClassifier()
    classifier.fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    return y_test, y_predicted

# def prepareData1():#https://www.kaggle.com/nphantawee/pump-sensor-data
#     labels = genfromtxt('sensor.csv', delimiter=',', dtype=None, usecols=(54), skip_header=1, converters={54: convert1})
#     data = genfromtxt('sensor.csv', delimiter=',', dtype=None, usecols=numpy.hstack([numpy.arange(2, 17), numpy.arange(18, 53)]), skip_header=1)
#
#     imp = SimpleImputer(missing_values=numpy.nan, strategy='mean')
#     data = imp.fit_transform(data)
#     numOfSelectedFeatures = 5
#     return labels, data, numOfSelectedFeatures

def prepareData2():#https://www.kaggle.com/cnic92/200-financial-indicators-of-us-stocks-20142018 (choose 2014)
    d = getConvertFuncs(58)
    data = genfromtxt('financialX.csv', delimiter=',', dtype=None, skip_header=1, converters=d)
    labels = genfromtxt('financialY.csv', delimiter=',', dtype=None, skip_header=1)
    print(numpy.where(numpy.isnan(data)))
    imp = SimpleImputer(missing_values=numpy.nan, strategy='mean')
    data = imp.fit_transform(data)
    numOfSelectedFeatures = 6
    return labels, data, numOfSelectedFeatures

def prepareData3():#https://www.kaggle.com/ivanloginov/the-broken-machine?select=xtrain.csv
    d = getConvertFuncs(58)
    data = genfromtxt('BrokenMachineX.csv', delimiter=',', dtype=None, skip_header=1, converters=d)
    labels = genfromtxt('BrokenMachineY.csv', delimiter=',', dtype=None, skip_header=1)
    imp = SimpleImputer(missing_values=numpy.nan, strategy='mean')
    data = imp.fit_transform(data)
    numOfSelectedFeatures = 6
    return labels, data, numOfSelectedFeatures

def prepareData4():#https://www.kaggle.com/dipayanbiswas/parkinsons-disease-speech-signal-features
    d = getConvertFuncs(754)
    labels = genfromtxt('pd_speech_features.csv', delimiter=',', dtype=None, usecols=(754), skip_header=1)
    data = genfromtxt('pd_speech_features.csv', delimiter=',', dtype=None,
                      usecols=numpy.hstack([numpy.arange(1, 753)]), skip_header=1, converters=d)

    imp = SimpleImputer(missing_values=numpy.nan, strategy='mean')
    data = imp.fit_transform(data)
    numOfSelectedFeatures = 75
    return labels, data, numOfSelectedFeatures

def prepareData5():#https://www.kaggle.com/balaka18/email-spam-classification-dataset-csv
    d = getConvertFuncs(3001)
    labels = genfromtxt('emails.csv', delimiter=',', dtype=None, usecols=(3001), skip_header=1)
    data = genfromtxt('emails.csv', delimiter=',', dtype=None,
                      usecols=numpy.hstack([numpy.arange(1, 3001)]), skip_header=1, converters=d)

    imp = SimpleImputer(missing_values=numpy.nan, strategy='mean')
    data = imp.fit_transform(data)
    numOfSelectedFeatures = 300
    return labels, data, numOfSelectedFeatures


def select_features(data: numpy.ndarray, labels: numpy.ndarray, k):
    selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_classif, k=k)
    data = selector.fit_transform(data, labels)
    return data


def main():
    labels, data, numOfSelectedFeatures = prepareData4()
    y_true, y_predicted = classify(data, labels)
    print("All features classification report")
    print(metrics.classification_report(y_true, y_predicted))
    data_selected = select_features(data, labels, numOfSelectedFeatures)
    y_true, y_predicted = classify(data_selected, labels)
    print("Selected features classification report")
    print(metrics.classification_report(y_true, y_predicted))

main()