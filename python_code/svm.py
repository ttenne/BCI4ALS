import scipy.io
from sklearn import svm
import numpy as np
from AR import getARCoefs
from SampEn import getSampEnCoefs
import matplotlib.pyplot as plt
from tabulate import tabulate

tag_dict = {
    3: 'idle ',
    1: 'left ',
    2: 'right'
}

def accuracy(y_test, y_pred, print_table=False):
    if print_table:
        print(tabulate(getResTable(y_test, y_pred)))
    return np.sum(y_test==y_pred)/y_test.shape[0]

def getResTable(y_test, y_pred):
    mat = [[0 for i in range(4)] for i in range(4)]
    mat[0] = ['', f'y_pred = {tag_dict[1]}', f'y_pred = {tag_dict[2]}', f'y_pred = {tag_dict[3]}']
    mat[1][0] = f'y = {tag_dict[1]}'
    mat[2][0] = f'y = {tag_dict[2]}'
    mat[3][0] = f'y = {tag_dict[3]}'
    for real_res in range(1,4):
        for pred_res in range(1,4):
            temp_pred = y_pred==pred_res
            temp_res = y_test==real_res
            mat[real_res][pred_res] = sum([1 for i, result in enumerate(temp_res) if result and temp_pred[i]])/sum(y_test==real_res)
    return mat

def svmPredict(path='C:\\Users\\yaels\\Desktop\\UnitedRecordings', lags=21, lags_starting_point=130, useSampEn=False):
    '''lags=21, lags_starting_point=130 based on validation set Sub20220821001-Sub20220821003'''
    MIData = scipy.io.loadmat(f'{path}\\MIData.mat')['MIData']
    #arrange train set
    y_train = scipy.io.loadmat(f'{path}\\LabelTrain.mat')['LabelTrain']
    y_train = np.reshape(y_train, -1)
    MIData_train = MIData[:len(y_train)]
    ARCoefsTensor = getARCoefs(MIData_train, lags, lags_starting_point)
    X_train = np.reshape(ARCoefsTensor, (ARCoefsTensor.shape[0],-1)) #reshape data to a matrix in a shape of (num_of_trials, total_feat_number)
    if useSampEn:
        SampEnMat = getSampEnCoefs(MIData_train)
        X_train = np.append(X_train, SampEnMat, axis=1)
    #arrange test set
    y_test = scipy.io.loadmat(f'{path}\\LabelTest.mat')['LabelTest']
    y_test = np.reshape(y_test, -1)
    MIData_test = MIData[len(y_train):]
    ARCoefsTensor = getARCoefs(MIData_test, lags, lags_starting_point)
    X_test = np.reshape(ARCoefsTensor, (ARCoefsTensor.shape[0],-1)) #reshape data to a matrix in a shape of (num_of_trials, total_feat_number)
    if useSampEn:
        SampEnMat = getSampEnCoefs(MIData_test)
        X_test = np.append(X_test, SampEnMat, axis=1)
    #predict
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, y_test

if __name__ == "__main__":
    y_pred, y_test = svmPredict()
    print(f'accuracy = {accuracy(y_test, y_pred, print_table=True)}')
