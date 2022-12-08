import scipy.io
from sklearn import svm
import numpy as np
from AR import getARCoefs
from SampEn import getSampEnCoefs
from ACSP import getACSPVars
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

def reshapeFeatures(features):
    if len(features.shape) == 3:
        return np.reshape(features, (features.shape[0],-1))
    elif len(features.shape) == 2:
        return features
    else:
        raise ValueError('Invalid dimenstion of feature tensor')

def addFeatures(oldFeatures, newFeatures):
    if len(oldFeatures) == 0:
        return reshapeFeatures(newFeatures)
    else:
        return np.append(oldFeatures, reshapeFeatures(newFeatures), axis=1)

def svmPredict(path='C:\\Users\\yaels\\Desktop\\UnitedRecordings', lags=21, lags_starting_point=130, useBS=False, useAR=True, useSampEn=False, r_val=0.2, useACSP=False, initial_var_trial_num=30, mu=0.95):
    '''lags=21, lags_starting_point=130 based on validation set Sub20220821001-Sub20220821003'''
    #fetch data
    MIData = scipy.io.loadmat(f'{path}\\MIData.mat')['MIData']
    y_train = scipy.io.loadmat(f'{path}\\LabelTrain.mat')['LabelTrain']
    y_train = np.reshape(y_train, -1)
    MIData_train = MIData[:len(y_train)]
    y_test = scipy.io.loadmat(f'{path}\\LabelTest.mat')['LabelTest']
    y_test = np.reshape(y_test, -1)
    MIData_test = MIData[len(y_train):]
    #arrange train and test set
    X_train = []
    X_test = []
    if useBS:
        BSFeatures = scipy.io.loadmat(f'{path}\\FeaturesTrainSelected.mat')['FeaturesTrainSelected']
        X_train = addFeatures(X_train, BSFeatures)
        BSFeatures = scipy.io.loadmat(f'{path}\\FeaturesTest.mat')['FeaturesTest']
        X_test = addFeatures(X_test, BSFeatures)
    if useAR:
        ARCoefsTensor = getARCoefs(MIData_train, lags, lags_starting_point)
        X_train = addFeatures(X_train, ARCoefsTensor)
        ARCoefsTensor = getARCoefs(MIData_test, lags, lags_starting_point)
        X_test = addFeatures(X_test, ARCoefsTensor)
    if useSampEn:
        SampEnMat = getSampEnCoefs(MIData_train, r_val)
        X_train = addFeatures(X_train, SampEnMat)
        SampEnMat = getSampEnCoefs(MIData_test, r_val)
        X_test = addFeatures(X_test, SampEnMat)
    if useACSP:
        CSPVarsMat_train, CSPVarsMat_test = getACSPVars(MIData_train, MIData_test, y_train, initial_var_trial_num, mu)
        X_train = addFeatures(X_train, CSPVarsMat_train)
        X_test = addFeatures(X_test, CSPVarsMat_test)
    #predict
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, y_test

if __name__ == "__main__":
    y_pred, y_test = svmPredict()
    print(f'accuracy = {accuracy(y_test, y_pred, print_table=True)}')
