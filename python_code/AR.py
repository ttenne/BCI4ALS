from statsmodels.tsa.ar_model import AutoReg
import numpy as np

def getARCoefs(MIData, lags):
    '''return a tensor of AR coefficients in the shape of (num_of_trials, num_of_electrodes, num_of_lags+1)'''
    coefs_tensor = []
    for trial in MIData:
        coefs_mat = []
        for electrode in trial:
            model = AutoReg(electrode, lags=lags).fit()
            coefs_vec = np.matrix(model.params)
            if len(coefs_mat) == 0:
                coefs_mat = coefs_vec
            else:
                coefs_mat = np.append(coefs_mat, coefs_vec, axis=0)
        coefs_mat = np.array([coefs_mat])
        if len(coefs_tensor) == 0:
            coefs_tensor = coefs_mat
        else:
            coefs_tensor = np.append(coefs_tensor, coefs_mat, axis=0)
    return coefs_tensor

# import scipy.io
# MIData = scipy.io.loadmat('C:\\Users\\yaels\\Desktop\\UnitedRecordings\\MIData.mat')['MIData']
# # model = AutoReg(MIData[0][0], lags=10).fit()
# # print(type(model.params))
# print(getARCoefs(MIData, 21).shape)
