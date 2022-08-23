function [] = MI4_trainTestSplit(recordingFolder)
%% This function makes the train-test split

%% Load previous variables:
targetLabels = cell2mat(struct2cell(load(strcat(recordingFolder,'/trainingVec'))));

numClasses = length(unique(targetLabels));                      % set number of possible targets (classes)
trialsForTest = 30;                                             % get number of trials for test set

%% Split data
idleIdx = find(targetLabels == 3);                                                      % find idle trials
testIdx = numClasses*length(idleIdx)-(trialsForTest-1):numClasses*length(idleIdx);      % take only last session as testset

% split test data
LabelTest = targetLabels(testIdx);          % taking the test trials labels from each class

% split train data
LabelTrain = targetLabels;
LabelTrain(testIdx) = [];                   % delete the test trials from the labels matrix, and keep only the train labels

% saving
save(strcat(recordingFolder,'/LabelTest.mat'),'LabelTest');
save(strcat(recordingFolder,'/LabelTrain.mat'),'LabelTrain');

end
