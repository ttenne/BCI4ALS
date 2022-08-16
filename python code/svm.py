import scipy.io
from sklearn import svm
import numpy as np

def accuracy(y_test, y_pred):
    return np.sum(y_test==y_pred)/y_test.shape[0]

# path = 'C:\\Recordings\\Sub20220811003'
path = 'C:\\Recordings\\Sub20220811merged'

# MIData = scipy.io.loadmat(f'{path}\\MIData.mat')['MIData']
# print(MIData.shape)
# exit()

X_train = scipy.io.loadmat(f'{path}\\FeaturesTrainSelected.mat')['FeaturesTrainSelected']
y_train = scipy.io.loadmat(f'{path}\\LabelTrain.mat')['LabelTrain']
y_train = np.reshape(y_train, -1)
X_test = scipy.io.loadmat(f'{path}\\FeaturesTest.mat')['FeaturesTest']
y_test = scipy.io.loadmat(f'{path}\\LabelTest.mat')['LabelTest']
y_test = np.reshape(y_test, -1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f'Accuarcy = {accuracy(y_test, y_pred)}')
print(y_pred)
print(y_test)