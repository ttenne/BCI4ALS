function [] = MI4_trainTestSplitRandom(recordingFolder)
%% This function makes the train-test split

%% Load previous variables:
targetLabels = cell2mat(struct2cell(load(strcat(recordingFolder,'/trainingVec'))));

numClasses = length(unique(targetLabels));                      % set number of possible targets (classes)

%% Split data
trials = size(targetLabels,2);                                        % get number of trials from main data variable
num4test = floor(0.2*trials/numClasses);                        % define how many test trials after feature extraction
idleIdx = find(targetLabels == 3);                  % find idle trials
leftIdx = find(targetLabels == 1);                  % find left trials
rightIdx = find(targetLabels == 2);                 % find right trials
testIdx = randperm(length(idleIdx),num4test);                       % picking test index randomly
testIdx = [idleIdx(testIdx) leftIdx(testIdx) rightIdx(testIdx)];    % taking the test index from each class
testIdx = sort(testIdx);                                            % sort the trials

% split test data
LabelTest = targetLabels(testIdx);          % taking the test trials labels from each class

% split train data
LabelTrain = targetLabels;
LabelTrain(testIdx) = [];                   % delete the test trials from the labels matrix, and keep only the train labels


% saving
save(strcat(recordingFolder,'/LabelTest.mat'),'LabelTest');
save(strcat(recordingFolder,'/LabelTrain.mat'),'LabelTrain');

end
