from statsmodels.tsa.ar_model import AutoReg
import numpy as np

def getARCoefs(MIData, lags, lags_starting_point):
    '''return a tensor of AR coefficients in the shape of (num_of_trials, num_of_electrodes, num_of_lags+1)'''
    coefs_tensor = []
    for trial in MIData:
        coefs_mat = []
        for electrode in trial:
            if lags_starting_point > 0:
                electrode = electrode[:-lags_starting_point]
            model = AutoReg(electrode, lags=lags).fit()
            coefs_vec = np.asarray(model.params)
            if len(coefs_mat) == 0:
                coefs_mat = coefs_vec
            else:
                coefs_mat = np.vstack([coefs_mat, coefs_vec])
        coefs_mat = np.array([coefs_mat])
        if len(coefs_tensor) == 0:
            coefs_tensor = coefs_mat
        else:
            coefs_tensor = np.append(coefs_tensor, coefs_mat, axis=0)
    return coefs_tensor
