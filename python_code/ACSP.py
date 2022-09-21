import numpy as np
import scipy.io
import scipy.linalg

def getCSPFilters(cov1, cov2):
    _, vr = scipy.linalg.eig(cov1, cov1+cov2)
    return vr

def calcCovMat(MIData):
    return sum([np.matmul(MIData[i], MIData[i].T) for i in range(len(MIData))])/len(MIData)

# def calcCovMatnumpy(MIData):
#     return sum([np.cov(MIData[i]) for i in range(len(MIData))])#/len(MIData)

def updateCovMat(cov_mat, trial, mu):
    return mu*cov_mat + (1-mu)*np.matmul(trial, trial.T)

def applyCSPFilters(MIData_test, filter1):
    for i in range(len(MIData_test)):
        MIData_test[i] = np.matmul(filter1.T, MIData_test[i])
    return MIData_test

def ACSP_OVR(MIData_train, y_train, initial_var_trial_num, mu):
    MIData_train1 = MIData_train[:initial_var_trial_num]
    y_train1 = y_train[:initial_var_trial_num]
    MIData_train2 = MIData_train[initial_var_trial_num:]
    y_train2 = y_train[initial_var_trial_num:]
    cov1 = calcCovMat(MIData_train1[y_train1==1])
    cov1_rest = calcCovMat(MIData_train[y_train!=1])
    cov2 = calcCovMat(MIData_train1[y_train1==2])
    cov2_rest = calcCovMat(MIData_train[y_train!=2])
    cov3 = calcCovMat(MIData_train[y_train==3])
    cov3_rest = calcCovMat(MIData_train[y_train!=3])
    for i, trial in enumerate(MIData_train2):
        if y_train2[i] == 1:
            cov1 = updateCovMat(cov1, trial, mu)
        elif y_train2[i] == 2:
            cov2 = updateCovMat(cov2, trial, mu)
        elif y_train2[i] == 3:
            cov3 = updateCovMat(cov3, trial, mu)
    filter1 = getCSPFilters(cov1, cov1_rest)
    filter2 = getCSPFilters(cov2, cov2_rest)
    filter3 = getCSPFilters(cov3, cov3_rest)
    return filter1, filter2, filter3

def getACSPVars(MIData_train, MIData_test, y_train, initial_var_trial_num, mu):
    filters = ACSP_OVR(MIData_train, y_train, initial_var_trial_num, mu)
    MIData_train_after_csp = [applyCSPFilters(MIData_train, filter) for filter in filters]
    MIData_test_after_csp = [applyCSPFilters(MIData_test, filter) for filter in filters]
    CSPVarsMat_train = []
    for proj in MIData_train_after_csp:
        proj_mat = []
        for trial in proj:
            vars_vec = []
            for electrode in trial:
                vars_vec.append(np.var(electrode))
            if len(proj_mat) == 0:
                proj_mat = np.asarray(vars_vec)
            else:
                vars_vec = np.asarray(vars_vec)
                proj_mat = np.vstack([proj_mat, vars_vec])
        if len(CSPVarsMat_train) == 0:
            CSPVarsMat_train = proj_mat
        else:
            CSPVarsMat_train = np.append(CSPVarsMat_train, proj_mat, axis=1)
    CSPVarsMat_test = []
    for proj in MIData_test_after_csp:
        proj_mat = []
        for trial in proj:
            vars_vec = []
            for electrode in trial:
                vars_vec.append(np.var(electrode))
            if len(proj_mat) == 0:
                proj_mat = np.asarray(vars_vec)
            else:
                vars_vec = np.asarray(vars_vec)
                proj_mat = np.vstack([proj_mat, vars_vec])
        if len(CSPVarsMat_test) == 0:
            CSPVarsMat_test = proj_mat
        else:
            CSPVarsMat_test = np.append(CSPVarsMat_test, proj_mat, axis=1)
    return CSPVarsMat_train, CSPVarsMat_test
