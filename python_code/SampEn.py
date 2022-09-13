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
            values_mat = np.asarray(values_vec)
        else:
            values_vec = np.asarray(values_vec)
            values_mat = np.vstack([values_mat, values_vec])
    return values_mat
