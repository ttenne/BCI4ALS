import scipy.io
from sklearn import svm
import numpy as np
from AR import getARCoefs
import matplotlib.pyplot as plt

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

def svmPredict(path, lags=21, print_table=False):
    '''lags=21 based validation set Sub20220821001-Sub20220821003'''
    MIData = scipy.io.loadmat(f'{path}\\MIData.mat')['MIData']

    y_train = scipy.io.loadmat(f'{path}\\LabelTrain.mat')['LabelTrain']
    y_train = np.reshape(y_train, -1)
    MIData_train = MIData[:len(y_train)]
    ARCoefsTensor = getARCoefs(MIData_train, lags)
    X_train = np.reshape(ARCoefsTensor, (ARCoefsTensor.shape[0],-1)) #reshape data to a matrix in a shape of (num_of_trials, total_feat_number)

    y_test = scipy.io.loadmat(f'{path}\\LabelTest.mat')['LabelTest']
    y_test = np.reshape(y_test, -1)
    MIData_test = MIData[len(y_train):]
    ARCoefsTensor = getARCoefs(MIData_test, lags)
    X_test = np.reshape(ARCoefsTensor, (ARCoefsTensor.shape[0],-1)) #reshape data to a matrix in a shape of (num_of_trials, total_feat_number)

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # exit()

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # print(y_pred)
    # print(y_test)
    if print_table:
        printTable(getResTable(y_test, y_pred))
    return accuracy(y_test, y_pred)

path = 'C:\\Users\\yaels\\Desktop\\UnitedRecordings'
print(svmPredict(path, print_table=True))

# accuracies = []
# max_lags = 100

# for lags in range(max_lags):
#     print(f'Working on score for lags = {lags}...')
#     accuracies.append(svmPredict(path, lags, True))

# plt.plot(range(max_lags), accuracies)
# plt.show()
