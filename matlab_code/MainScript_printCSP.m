%% MI Offline Main Script
% This script runs all the steps in order. Training -- Pre-processing --
% Data segmentation -- Feature extraction -- Model training.
% Two important points:
% 1. Remember the ID number (without zero in the beginning) for each different person
% 2. Remember the Lab Recorder filename and folder.

% Prequisites: Refer to the installation manual for required softare tools.

% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.


clc; clear; close all;

%% Run stimulation and record EEG data
%[recordingFolder] = MI1_offline_training();
%disp('Finished stimulation and EEG recording. Stop the LabRecorder and press any key to continue...');
%pause;

recordingsFolder = 'C:\Users\yaels\Desktop\Recordings\';
listdir = dir(recordingsFolder);
dir_names = {listdir().name};
for i = 1:length(dir_names)
    dir_names(i) = strcat(recordingsFolder, dir_names(i));
end

start_dir = strcat(recordingsFolder,'Sub20220821003');
end_dir = strcat(recordingsFolder,'Sub20220821003');

filtered_dir_names = {};
should_append = false;
count = 1;
for i = 1:length(dir_names)
    if strcmp(dir_names(i), start_dir)
        should_append = true;
    end
    if should_append == true
        filtered_dir_names(count) = dir_names(i);
        count = count+1;
    end
    if strcmp(dir_names(i), end_dir)
        should_append = false;
    end
end

for i = 1:length(filtered_dir_names)
    recordingFolder = char(filtered_dir_names(i));
    MI2_preprocess(recordingFolder);
    MI3_segmentation(recordingFolder);
    MI4_printCSP(recordingFolder);
    fprintf("Finished printing %s\n", recordingFolder);
    pause;
end
