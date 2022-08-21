import scipy.io
from sklearn import svm
import numpy as np

tag_dict = {
    3: 'idle ',
    1: 'left ',
    2: 'right'
}

def accuracy(y_test, y_pred):
    return np.sum(y_test==y_pred)/y_test.shape[0]

def getResTable(y_test, y_pred):
    mat = [[0 for i in range(4)] for i in range(4)]
    mat[0] = ['', tag_dict[1], tag_dict[2], tag_dict[3]]
    mat[1][0] = tag_dict[1]
    mat[2][0] = tag_dict[2]
    mat[3][0] = tag_dict[3]
    for real_res in range(1,4):
        for pred_res in range(1,4):
            temp_pred = y_pred==pred_res
            temp_res = y_test==real_res
            mat[real_res][pred_res] = sum([1 for i, result in enumerate(temp_res) if result and temp_pred[i]])/sum(y_test==real_res)
    return mat

def printTable(mat):
    for row in mat:
        for item in row:
            print(item, end="     ")
        print('')

# path = 'C:\\Recordings\\Sub20220811003'
path = 'C:\\Users\\yaels\\Desktop\\UnitedRecordings'

X_train = scipy.io.loadmat(f'{path}\\FeaturesTrainSelected.mat')['FeaturesTrainSelected']
y_train = scipy.io.loadmat(f'{path}\\LabelTrain.mat')['LabelTrain']
y_train = np.reshape(y_train, -1)
X_test = scipy.io.loadmat(f'{path}\\FeaturesTest.mat')['FeaturesTest']
y_test = scipy.io.loadmat(f'{path}\\LabelTest.mat')['LabelTest']
y_test = np.reshape(y_test, -1)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f'Accuarcy = {accuracy(y_test, y_pred)}')
# print(y_pred)
# print(y_test)
printTable(getResTable(y_test, y_pred))