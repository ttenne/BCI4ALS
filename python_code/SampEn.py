from EntropyHub import SampEn
import numpy as np

def getSampEnCoefs(MIData):
    '''return a matrix of SampEn values in the shape of (num_of_trials, num_of_electrodes)'''
    values_mat = []
    for trial in MIData:
        values_vec = []
        for electrode in trial:
            Samp, _, _ = SampEn(electrode)
            values_vec.append(Samp[-1])
        if len(values_mat) == 0:
            values_mat = np.matrix(values_vec)
        else:
            values_vec = np.matrix(values_vec)
            values_mat = np.append(values_mat, values_vec, axis=0)
    return values_mat

import scipy.io
path2 = 'C:\\Users\\yaels\\Desktop\\UnitedRecordings'
MIData = scipy.io.loadmat(f'{path2}\\MIData.mat')['MIData']

print(getSampEnCoefs(MIData).shape)
