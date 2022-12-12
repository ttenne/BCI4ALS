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

# def applyCSPFilters(MIData_test, filter1):
#     for i in range(len(MIData_test)):
#         MIData_test[i] = np.matmul(filter1.T, MIData_test[i])
#     return MIData_test

def applyCSPFilter(MIData_trial, filter1):
    return np.matmul(filter1.T, MIData_trial)

def ACSP_OVR_init(MIData_train, y_train, initial_var_trial_num, num_of_classes=3):
    MIData_train_init = MIData_train[:initial_var_trial_num]
    y_train_init = y_train[:initial_var_trial_num]
    cov_mat_list = [calcCovMat(MIData_train_init[y_train_init==i]) for i in range(1,1+num_of_classes)]
    cov_rest_mat_list = [calcCovMat(MIData_train_init[y_train_init!=i]) for i in range(1,1+num_of_classes)]
    filter_list = [getCSPFilters(cov_mat_list[i], cov_rest_mat_list[i]) for i in range(num_of_classes)]
    return filter_list, cov_mat_list, cov_rest_mat_list

def ACSP_OVR_update(MIData_train_trial, y_train_trial, mu, filter_list, cov_mat_list, cov_rest_mat_list):
    cov_mat_list[y_train_trial-1] = updateCovMat(cov_mat_list[y_train_trial-1], MIData_train_trial, mu)
    cov_rest_mat_list = [updateCovMat(cov_rest_mat, MIData_train_trial, mu) if i != y_train_trial-1 else cov_rest_mat for i, cov_rest_mat in enumerate(cov_rest_mat_list)]
    filter_list = [getCSPFilters(cov_mat_list[i], cov_rest_mat_list[i]) for i in range(len(filter_list))]
    return filter_list, cov_mat_list, cov_rest_mat_list

def ACSP_OVR(MIData_train, y_train, MIData_test, initial_var_trial_num, mu):
    filter_list, cov_mat_list, cov_rest_mat_list = ACSP_OVR_init(MIData_train, y_train, initial_var_trial_num)
    MIData_train_after_csp = [[] for i in range(len(filter_list))]
    MIData_test_after_csp = [[] for i in range(len(filter_list))]
    for trial, MIData_train_trial in enumerate(MIData_train):
        if trial >= initial_var_trial_num:
            filter_list, cov_mat_list, cov_rest_mat_list = ACSP_OVR_update(MIData_train_trial, y_train[trial], mu, filter_list, cov_mat_list, cov_rest_mat_list)
        for i, filter1 in enumerate(filter_list):
            MIData_train_after_csp[i].append(applyCSPFilter(MIData_train_trial, filter1))
    for trial, MIData_test_trial in enumerate(MIData_test):
        filter_list, cov_mat_list, cov_rest_mat_list = ACSP_OVR_update(MIData_train_trial, y_train[trial], mu, filter_list, cov_mat_list, cov_rest_mat_list)
        for i, filter1 in enumerate(filter_list):
                MIData_test_after_csp[i].append(applyCSPFilter(MIData_test_trial, filter1))
    return MIData_train_after_csp, MIData_test_after_csp

def getACSPVars(MIData_train, MIData_test, y_train, initial_var_trial_num, mu):
    MIData_train_after_csp, MIData_test_after_csp = ACSP_OVR(MIData_train, y_train, MIData_test, initial_var_trial_num, mu)
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
