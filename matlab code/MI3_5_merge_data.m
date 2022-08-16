function MI3_5_merge_data(recordingFolder)
recordingFolder1 = 'C:\Recordings\Sub20220811002';
recordingFolder2 = 'C:\Recordings\Sub20220811003';
numTrials = 60;

%merge trainig vectors
trainingVec1 = load(strcat(recordingFolder1,'/trainingVec.mat'));
trainingVec2 = load(strcat(recordingFolder2,'/trainingVec.mat'));

trainingVec1 = cell2mat(struct2cell(trainingVec1));
trainingVec2 = cell2mat(struct2cell(trainingVec2));

if size(trainingVec1,2) ~= size(trainingVec2,2) && size(trainingVec1,3) ~= size(trainingVec2,3)
    disp('training vector files not of same size!');
else
    trainingVec = zeros(1,length(trainingVec1));
    for trial = 1:length(trainingVec1)
        trainingVec(trial) = trainingVec1(trial);
    end
    for trial = length(trainingVec1)+1:length(trainingVec1)+length(trainingVec2)
        trainingVec(trial) = trainingVec2(trial-length(trainingVec1));
    end
    %size(trainingVec)
    save(strcat(recordingFolder,'/trainingVec.mat'),'trainingVec');
end

%merge MIData
data1 = load(strcat(recordingFolder1,'/MIData.mat'));
data2 = load(strcat(recordingFolder2,'/MIData.mat'));

matrix1 = cell2mat(struct2cell(data1));
matrix2 = cell2mat(struct2cell(data2));

if size(matrix1,1) ~= numTrials
    disp('File 1 corrupted!');
    fprintf('File 1 has %d trials instead of %d\n', size(matrix1,1), numTrials);
end

if size(matrix2,1) ~= numTrials
    disp('File 2 corrupted!');
    fprintf('File 2 has %d trials instead of %d\n', size(matrix2,1), numTrials);
end

if size(matrix1,2) ~= size(matrix2,2) && size(matrix1,3) ~= size(matrix2,3)
    disp('MIData files not of same size!');
else
    MIData = zeros(length(matrix1(:,1,1))+length(matrix2(:,1,1)),length(matrix1(1,:,1)),length(matrix1(1,1,:)));
    for trial = 1:length(matrix1(:,1,1))
        MIData(trial,:,:) = matrix1(trial,:,:);
    end
    for trial = length(matrix1(:,1,1))+1:length(matrix1(:,1,1))+length(matrix2(:,1,1))
        MIData(trial,:,:) = matrix2(trial-length(matrix1(:,1,1)),:,:);
    end
    %size(MIData)
    save(strcat(recordingFolder,'/MIData.mat'),'MIData');
end

%copy additional files
copyfile(strcat(recordingFolder1,'/EEG_chans.mat'), recordingFolder);

end
