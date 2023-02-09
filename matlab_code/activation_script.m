clc; clear; close all;
recordingsFolder = 'C:\Users\yaels\Desktop\Recordings\';
start_dir = strcat(recordingsFolder,'Sub20220821001');
end_dir = strcat(recordingsFolder,'Sub20220823001');

%MainScript_light(recordingsFolder, start_dir, end_dir);
%MainScript_light_BSFeatures(recordingsFolder, start_dir, end_dir);
%MainScript_lightRandom(recordingsFolder, start_dir, end_dir);
%MainScript_lightRandom_BSFeatures(recordingsFolder, start_dir, end_dir);
MainScript_ultraLight(recordingsFolder, start_dir, end_dir);
%MainScript_ultraLight_BSFeatures(recordingsFolder, start_dir, end_dir);
