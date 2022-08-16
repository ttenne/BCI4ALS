function MI3_5_merge_data(recordingFolder, filtered_dir_names)
numTrials = 60;

%merge trainig vectors
tempTrainingVec1 = load(strcat(char(filtered_dir_names(1)),'/trainingVec.mat'));
tempTrainingVec1 = cell2mat(struct2cell(tempTrainingVec1));

trainingVec = [];
current_trial = 0;

for i = 1:length(filtered_dir_names)
    tempTrainingVec = load(strcat(char(filtered_dir_names(i)),'/trainingVec.mat'));
    tempTrainingVec = cell2mat(struct2cell(tempTrainingVec));
    if size(tempTrainingVec1,2) ~= size(tempTrainingVec,2) && size(tempTrainingVec1,3) ~= size(tempTrainingVec,3)
        fprintf('training vector from %s is not the same size as training vector from %s\n', char(filtered_dir_names(i)), char(filtered_dir_names(1)));
    else
        for trial = 1:length(tempTrainingVec)
            trainingVec(trial+current_trial) = tempTrainingVec(trial);
        end
        current_trial = current_trial + length(tempTrainingVec);
    end
end
save(strcat(recordingFolder,'/trainingVec.mat'),'trainingVec');

%merge MIData
tempMIData1 = load(strcat(char(filtered_dir_names(1)),'/MIData.mat'));
tempMIData1 = cell2mat(struct2cell(tempMIData1));

MIData = [];
current_trial = 0;

for i = 1:length(filtered_dir_names)
    tempMIData = load(strcat(char(filtered_dir_names(i)),'/MIData.mat'));
    tempMIData = cell2mat(struct2cell(tempMIData));
    if size(tempMIData,1) ~= numTrials
        fprintf('%s corrupted!\n', strcat(char(filtered_dir_names(i)),'/MIData.mat'));
        fprintf('%s has %d trials instead of %d\n', strcat(char(filtered_dir_names(i)),'/MIData.mat'), size(tempMIData,1), numTrials);
    end
    if size(tempMIData1,2) ~= size(tempMIData,2) && size(tempMIData1,3) ~= size(tempMIData,3)
        fprintf('MIData from %s is not the same size as MIData from %s\n', char(filtered_dir_names(i)), char(filtered_dir_names(1)));
    else
        for trial = 1:length(tempMIData(:,1,1))
            MIData(trial+current_trial,:,:) = tempMIData(trial,:,:);
        end
        current_trial = current_trial + length(tempMIData(:,1,1));
    end
end
save(strcat(recordingFolder,'/MIData.mat'),'MIData');

%copy additional files
copyfile(strcat(char(filtered_dir_names(1)),'/EEG_chans.mat'), recordingFolder);

end
