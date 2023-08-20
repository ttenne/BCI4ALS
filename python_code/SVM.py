import scipy.io
from sklearn import svm
import numpy as np
from AR import getARCoefs
from SampEn import getSampEnCoefs
from ACSP import getACSPVars
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.feature_selection import mutual_info_classif
from AutoEnc import AutoEncoder
from GAN import GAN
from TimeGAN import TimeGANAux

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

def printSelectedFeatures(selected_features, useBS, useAR, useSampEn, useACSP):
    cur_index = 0
    BS_indices = list(range(cur_index, 10 if useBS else 0))
    cur_index += len(BS_indices)
    AR_indices = list(range(cur_index, cur_index + 242 if useAR else cur_index))
    cur_index += len(AR_indices)
    SampEn_indices = list(range(cur_index, cur_index + 11 if useSampEn else cur_index))
    cur_index += len(SampEn_indices)
    ACSP_indices = list(range(cur_index, cur_index + 33 if useACSP else cur_index))
    BS_count = 0
    AR_count = 0
    SampEn_count = 0
    ACSP_count = 0
    for selected_feature in selected_features:
        if selected_feature in BS_indices:
            BS_count += 1
        elif selected_feature in AR_indices:
            AR_count += 1
        elif selected_feature in SampEn_indices:
            SampEn_count += 1
        elif selected_feature in ACSP_indices:
            ACSP_count += 1
        else:
            raise ValueError()
    if useBS:
        print(f'BS_count = {BS_count}')
    if useAR:
        print(f'AR_count = {AR_count}')
    if useSampEn:
        print(f'SampEn_count = {SampEn_count}')
    if useACSP:
        print(f'ACSP_count = {ACSP_count}')

def fetchData(path, useAutoEnc, useGAN, num_of_gen_batches):
    batch_size = None
    MIData = scipy.io.loadmat(f'{path}\\MIData.mat')['MIData']
    y_train = scipy.io.loadmat(f'{path}\\LabelTrain.mat')['LabelTrain']
    y_train = np.reshape(y_train, -1)
    MIData_train = MIData[:len(y_train)]
    y_test = scipy.io.loadmat(f'{path}\\LabelTest.mat')['LabelTest']
    y_test = np.reshape(y_test, -1)
    MIData_test = MIData[len(y_train):]
    if useGAN and num_of_gen_batches > 0:
        batch_size = len(MIData_train)//3
        MIData_train_syn = []
        for label in [1,2,3]:
            MIData_train_temp = MIData_train[y_train == label]
            gan = TimeGANAux(train_data=MIData_train_temp, label=label, batch_size=batch_size)
            gan.train()
            MIData_train_syn.append(np.array(gan.generate(num_of_gen_batches), dtype=np.float64))
        MIData_train_syn = np.array(MIData_train_syn)
        MIData_train_syn = np.reshape(MIData_train_syn, (MIData_train_syn.shape[0]*MIData_train_syn.shape[1], MIData_train_syn.shape[2], MIData_train_syn.shape[3]))
        y_train_syn = np.array([1 + i//(MIData_train_syn.shape[0]//3) for i in range(len(MIData_train_syn))])
        perm = np.random.permutation(len(MIData_train_syn))
        MIData_train_syn = MIData_train_syn[perm]
        y_train_syn = y_train_syn[perm]
        np.random.shuffle(MIData_train_syn)
        MIData_train = np.concatenate((MIData_train, MIData_train_syn))
        y_train = np.concatenate((y_train, y_train_syn))
    if useAutoEnc:
        print('Creating auto-encoder CNN...')
        auto_encoder = AutoEncoder(MIData_train)#, epochs=10)
        print('Filtering MIData...')
        MIData_train = auto_encoder.predict(MIData_train)
        MIData_test = auto_encoder.predict(MIData_test)
    return MIData_train, MIData_test, y_train, y_test, batch_size

def svmPredict(path='C:\\Users\\yaels\\Desktop\\UnitedRecordings', lags=21, lags_starting_point=130, useBS=False, useAR=False, useSampEn=False, r_val=0.2,
               useACSP=False, initial_var_trial_num=20, mu=0.95, useFeatSelAlg=False, num_of_selected_features=250, print_selected_features=False,
               useAutoEnc=False, useGAN=False, num_of_gen_batches=1):
    '''lags=21, lags_starting_point=130 based on validation set Sub20220821001-Sub20220821003'''
    if useBS and useGAN:
        raise ValueError("Can't use both useBS and useGan")
    #fetch data
    print('Fetching MIData...')
    MIData_train, MIData_test, y_train, y_test, batch_size = fetchData(path, useAutoEnc, useGAN, num_of_gen_batches)
    #arrange train and test set
    X_train = []
    X_test = []
    if useBS:
        print('Extracting BS features...')
        BSFeatures = scipy.io.loadmat(f'{path}\\FeaturesTrainSelected.mat')['FeaturesTrainSelected']
        X_train = addFeatures(X_train, BSFeatures)
        BSFeatures = scipy.io.loadmat(f'{path}\\FeaturesTest.mat')['FeaturesTest']
        X_test = addFeatures(X_test, BSFeatures)
    if useAR:
        print('Extracting AR features...')
        ARCoefsTensor = getARCoefs(MIData_train, lags, lags_starting_point)
        X_train = addFeatures(X_train, ARCoefsTensor)
        ARCoefsTensor = getARCoefs(MIData_test, lags, lags_starting_point)
        X_test = addFeatures(X_test, ARCoefsTensor)
    if useSampEn:
        print('Extracting SampEn features...')
        SampEnMat = getSampEnCoefs(MIData_train, r_val)
        X_train = addFeatures(X_train, SampEnMat)
        SampEnMat = getSampEnCoefs(MIData_test, r_val)
        X_test = addFeatures(X_test, SampEnMat)
    if useACSP:
        print('Extracting ACSP features...')
        CSPVarsMat_train, CSPVarsMat_test = getACSPVars(MIData_train, MIData_test, y_train, initial_var_trial_num, mu)
        X_train = addFeatures(X_train, CSPVarsMat_train)
        X_test = addFeatures(X_test, CSPVarsMat_test)
    #apply feature selection
    if useFeatSelAlg:
        print('Executing feature selection...')
        MI_values = mutual_info_classif(X_train, y_train, discrete_features=False)
        # # validate number of selected features - this is here and not in the validation.py file, because we want to reuse the exact same features for each iteration so using the svmPredict function every time is a huge waste of time
        # accuracies = []
        # init_feat_num = 10
        # num_of_selected_features_list = list(range(init_feat_num, X_train.shape[-1]+1))
        # sorted_features = sorted(range(len(MI_values)), key = lambda sub: MI_values[sub])
        # for num_of_selected_features in num_of_selected_features_list:
        #     print(f'running with {num_of_selected_features} features...')
        #     selected_features = sorted_features[-num_of_selected_features:]
        #     X_train_selected = X_train[:, selected_features]
        #     X_test_selected = X_test[:, selected_features]
        #     clf = svm.SVC()
        #     clf.fit(X_train_selected, y_train)
        #     y_pred = clf.predict(X_test_selected)
        #     accuracies.append(accuracy(y_test, y_pred))
        # # plt.plot(num_of_selected_features_list, accuracies)
        # # plt.xlabel('num_of_selected_features')
        # # plt.ylabel('validation score')
        # # plt.savefig(f'validateMIFeatSel with {num_of_gen_batches * 30} TimeGAN trials.png')
        # # plt.clf()
        # num_synth_feat = 0 if batch_size is None else num_of_gen_batches * batch_size
        # with open(f'{num_synth_feat} synth features accuracies.txt', 'w') as fp:
        #     fp.writelines((str(num_of_selected_features_list)+'\n', str(accuracies)+'\n'))
        # print(f'max accuracy = {np.max(accuracies)}')
        # print(f'num_of_selected_features = {init_feat_num + np.argmax(accuracies)}')
        # return
        selected_features = sorted(range(len(MI_values)), key = lambda sub: MI_values[sub])[-num_of_selected_features:] #extract num_of_selected_features features with best MI value
        if print_selected_features:
            printSelectedFeatures(selected_features, useBS, useAR, useSampEn, useACSP)
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]
    else:
        X_train_selected = X_train
        X_test_selected = X_test
    #predict
    clf = svm.SVC()
    print('Training SVM...')
    clf.fit(X_train_selected, y_train)
    print('Executing classification...')
    y_pred = clf.predict(X_test_selected)
    return y_pred, y_test
