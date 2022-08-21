function [] = MI4_featureExtraction(recordingFolder)
%% This function extracts features for the machine learning process.
% Starts by visualizing the data (power spectrum) to find the best powerbands.
% Next section computes the best common spatial patterns from all available
% labeled training trials. The next part extracts all learned features.
% This includes a non-exhaustive list of possible features (commented below).
% At the bottom there is a simple feature importance test that chooses the
% best features and saves them for model training.


%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.

%% Load previous variables:
load(strcat(recordingFolder,'/EEG_chans.mat'));                  % load the openBCI channel location
load(strcat(recordingFolder,'/MIData.mat'));                     % load the EEG data
targetLabels = cell2mat(struct2cell(load(strcat(recordingFolder,'/trainingVec'))));

Features2Select = 10;                                           % number of featuers for feature selection
numClasses = length(unique(targetLabels));                      % set number of possible targets (classes)
Fs = 125;                                                       % openBCI Cyton+Daisy by Bluetooth sample rate
trials = size(MIData,1);                                        % get number of trials from main data variable
num4test = floor(0.2*trials/numClasses);                        % define how many test trials after feature extraction
[R, C] = size(EEG_chans);                                       % get EEG_chans (char matrix) size - rows and columns
chanLocs = reshape(EEG_chans',[1, R*C]);                        % reshape into a vector in the correct order
numChans = size(MIData,2);                                      % get number of channels from main data variable

%% Common Spatial Patterns
% create a spatial filter using available EEG & labels
% we will "train" a mixing matrix (wTrain) on 80% of the trials and another
% mixing matrix (wViz) just for the visualization trial (vizTrial). This
% serves to show an understandable demonstration of the process.

% Begin by splitting into two classes:
leftClass = MIData(targetLabels == 1,:,:);
rightClass = MIData(targetLabels == 2,:,:);

% Aggregate all trials into one matrix
overallLeft = [];
overallRight = [];
idleIdx = find(targetLabels == 3);                  % find idle trials
leftIdx = find(targetLabels == 1);                  % find left trials
rightIdx = find(targetLabels == 2);                 % find right trials
rightIndices = rightIdx(randperm(length(rightIdx)));% randomize right indexs
leftIndices  = leftIdx(randperm(length(leftIdx)));   % randomize left indexs
idleIndices  = idleIdx(randperm(length(idleIdx)));   % randomize idle indexs
minTrials = min([length(leftIndices), length(rightIndices)]);
percentIdx = floor(0.8*minTrials);                  % this is the 80% part...
for trial=1:percentIdx
    overallLeft = [overallLeft squeeze(MIData(leftIndices(trial),:,:))];
    overallRight = [overallRight squeeze(MIData(rightIndices(trial),:,:))];
end

% visualize the CSP data:
%vizTrial = 9;      % cherry-picked!
for vizTrial = 1:20
    figure;
    subplot(1,2,1)      % show a single trial before CSP seperation
    scatter3(squeeze(leftClass(vizTrial,1,:)),squeeze(leftClass(vizTrial,2,:)),squeeze(leftClass(vizTrial,3,:)),'b'); hold on
    scatter3(squeeze(rightClass(vizTrial,1,:)),squeeze(rightClass(vizTrial,2,:)),squeeze(rightClass(vizTrial,3,:)),'g');
    title('Before CSP')
    legend('Left','Right')
    xlabel('channel 1')
    ylabel('channel 2')
    zlabel('channel 3')
    % find mixing matrix (wAll) for all trials:
    [wTrain, lambda, A] = csp(overallLeft, overallRight);
    % find mixing matrix (wViz) just for visualization trial:
    [wViz, lambdaViz, Aviz] = csp(squeeze(rightClass(vizTrial,:,:)), squeeze(leftClass(vizTrial,:,:)));
    % apply mixing matrix on available data (for visualization)
    leftClassCSP = (wViz'*squeeze(leftClass(vizTrial,:,:)));
    rightClassCSP = (wViz'*squeeze(rightClass(vizTrial,:,:)));

    subplot(1,2,2)      % show a single trial after CSP seperation
    scatter3(squeeze(leftClassCSP(1,:)),squeeze(leftClassCSP(2,:)),squeeze(leftClassCSP(3,:)),'b'); hold on
    scatter3(squeeze(rightClassCSP(1,:)),squeeze(rightClassCSP(2,:)),squeeze(rightClassCSP(3,:)),'g');
    title('After CSP')
    legend('Left','Right')
    xlabel('CSP dimension 1')
    ylabel('CSP dimension 2')
    zlabel('CSP dimension 3')

    clear leftClassCSP rightClassCSP Wviz lambdaViz Aviz
    pause;
end

