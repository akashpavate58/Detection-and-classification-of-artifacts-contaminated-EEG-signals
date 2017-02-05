%% Run detect on artifact data

close all;
clear;
clc;

%% Constants

%Right now it's set up so that each trial is one second worth of data; data recorded at 250Hz
cLengthOfTrial_Seconds = 1;

cSampleRate = 250;

cNumberOfSamplesPerTrial = cLengthOfTrial_Seconds * cSampleRate;

% Channel 2 was bad
cNumberOfChannels = 11;

%anonymous function hack to get the name of TrialData
%without hard coding it
%http://stackoverflow.com/questions/6681798/print-variable-name-in-matlab
varname=@(x) inputname(1);

cNumberOfTrialsPerNoiseType = 10;

cNumberOfNoiseTypes = 2;

cNumberOfTrainingTrials = cNumberOfTrialsPerNoiseType * cNumberOfNoiseTypes;


%% Load each file into a single training data matrix

TrainingData = zeros(cNumberOfChannels, cNumberOfSamplesPerTrial, cNumberOfTrainingTrials);

SelectedTrainingDirectory = uigetdir();

FolderStructsInDir = dir(SelectedTrainingDirectory);

NumberOfItemsInDir = length(FolderStructsInDir);

Labels = cell(1, cNumberOfTrainingTrials);

TrainingTrialIndex = 1;

for FolderIndex = 1 : NumberOfItemsInDir
    
    CurrentFolderStruct = FolderStructsInDir(FolderIndex);
    
    CurrentFolderName = FolderStructsInDir(FolderIndex).name;
    
    [~, ~, CurrentItemExt] = fileparts(CurrentFolderName);
    
    %Don't look at anything other than folders
    if(~strcmp(CurrentItemExt,''))
        continue;
    end
    
    %Ignore the garbage file names returned by dir
    if(strcmp(CurrentFolderName, '.'))
        continue;
        
    elseif(strcmp(CurrentFolderName, '..') )
        continue;
        
    end
    
    
    CurrentFolderPath = [SelectedTrainingDirectory, '\', CurrentFolderName];
    
    FileStructsInDir = dir(CurrentFolderPath);
    
    NumberOfItemsInFilesList = length(FileStructsInDir);
    
    
    for FileIndex = 1 : NumberOfItemsInFilesList
        
        TrainingTrialFileName = FileStructsInDir(FileIndex).name;
        TrainingTrialFilePath = [CurrentFolderPath, '\', TrainingTrialFileName];
        
        %Ignore the garbage file names returned by dir
        if( strcmp(TrainingTrialFileName, '.'))
            continue;
            
        elseif( strcmp(TrainingTrialFileName, '..') )
            continue;
            
        end
        
        [~, ~, CurrentFileExt] = fileparts(TrainingTrialFileName);
        
        %note that folders are just shown as files without an extension
        if(~strcmp(CurrentFileExt, '.mat'))
            continue;
            
        end
        
        
        [~, TrainingTrialFileName_NoExt, ~] = fileparts(TrainingTrialFileName);
        
        TrainingFileStruct = load(TrainingTrialFilePath);
        
        %It's assumed that the data will be stored in a field in the struct
        %where the field is the same name as the file, without the
        %extension
        TrainingFileData = TrainingFileStruct.(TrainingTrialFileName_NoExt);
        
        TrainingData(1, :, TrainingTrialIndex) = TrainingFileData(1, :);
        %                                       Skipping channel 2 (bad data)
        TrainingData(2, :, TrainingTrialIndex) = TrainingFileData(3, :);
        TrainingData(3, :, TrainingTrialIndex) = TrainingFileData(4, :);
        TrainingData(4, :, TrainingTrialIndex) = TrainingFileData(5, :);
        TrainingData(5, :, TrainingTrialIndex) = TrainingFileData(6, :);
        TrainingData(6, :, TrainingTrialIndex) = TrainingFileData(7, :);
        TrainingData(7, :, TrainingTrialIndex) = TrainingFileData(8, :);
        TrainingData(8, :, TrainingTrialIndex) = TrainingFileData(9, :);
        TrainingData(9, :, TrainingTrialIndex) = TrainingFileData(10, :);
        TrainingData(10, :, TrainingTrialIndex) = TrainingFileData(11, :);
        TrainingData(11, :, TrainingTrialIndex) = TrainingFileData(12, :);
        
        for ChannelIndex = 1 : cNumberOfChannels
            TrainingData(ChannelIndex, :, TrainingTrialIndex) = TrainingFileData(ChannelIndex,:);
        end
        %The name of the label is just the folder it came from
        Labels{TrainingTrialIndex} = CurrentFolderName;
        
        TrainingTrialIndex = TrainingTrialIndex + 1;
        
    end
    
end

perm = randperm(70,49);
TRAIN = TrainingData(:,:,perm);
TrainingTrialSet = pop_importdata(     'setname',  'Training Data Set', ...
    'data',     varname(TRAIN), ...
    'nbchan',   cNumberOfChannels, ...
    'srate',    cSampleRate, ...
    'pnts',      cNumberOfSamplesPerTrial, ...
    'comments', ['Training Data Loaded from folder ',SelectedTrainingDirectory]);

TRAINLABEL = Labels(1,perm);
DETECTModel = getModel(TrainingTrialSet,TRAINLABEL );



%% Test the DETECT model

% [TestingTrialFileName, TestingTrialFolder, ~] = uigetfile('*.mat', 'Select a Testing trial data file to run DETECT on.');
%
% TestingTrialFilePath = [TestingTrialFolder, '\', TestingTrialFileName];
%
% [~, TestingTrialFileName_NoExt, ~] = fileparts(TestingTrialFileName);
%
% TestingTrialStruct = load(TestingTrialFilePath);
%
% %It's assumed that the data will be stored in a field in the struct
% %where the field is the same name as the file, without the
% %extension
% TestingFileData = TestingTrialStruct.(TestingTrialFileName_NoExt);
%
% TestingData = zeros(cNumberOfChannels, cNumberOfSamplesPerTrial);
%
% TestingData(1, :) = TestingFileData(1, :);
% %            Skipping channel 2 (bad data)
% TestingData(2, :) = TestingFileData(3, :);
% TestingData(3, :) = TestingFileData(4, :);
% TestingData(4, :) = TestingFileData(5, :);
% TestingData(5, :) = TestingFileData(6, :);
% TestingData(6, :) = TestingFileData(7, :);
% TestingData(7, :) = TestingFileData(8, :);
% TestingData(8, :) = TestingFileData(9, :);
% TestingData(9, :) = TestingFileData(10, :);
% TestingData(10, :) = TestingFileData(11, :);
% TestingData(11, :) = TestingFileData(12, :);
%


testIdx = setdiff(1:70,perm);
TEST = TrainingData(:,:,testIdx);
TESTlabels = Labels(1,testIdx);

CR = [];

for i=1:size(TESTlabels,2)
    TestSample = TEST(:,:,i);
    
    TestingTrialSet = pop_importdata(  'setname',  'Test Data Set', ...
        'data',     varname(TestSample), ...
        'nbchan',   cNumberOfChannels, ...
        'srate',    cSampleRate, ...
        'comments', 'Testing Data Loaded using RunDetect_ArtifactData' );
    
    
    
    
    %Note: I'm setting slideWidth to .250, (.250 seconds) because I'm thinking that each trial
    %contains one instance of noise.
    
    % disp('Labeling data to test.... (this may take some time)');
    
    % in getARFeatures.m, I get 'Model does not support probabiliy estimates',
    % which seems to be coming from svmpredict_DETECT.c
    
    % fprintf('the label of sample is %s \n\n', TESTlabels{1,i});
    Results = labelData(TestingTrialSet, DETECTModel, 250, .250);
    % fprintf('the label of prediction is %s', Results.label);
    
    CR(i)=strcmp(TESTlabels{1,i},Results.label);
end

Res = sum(CR)/21;
fprintf('The Accuracy is %2.3f', Res);

% this would be nice to use, but I get
%   The name 'Units' is not an accessible property
%   for an instance of class 'matlab.graphics.GraphicsPlaceholder'.

% For some reason eegplot2, and even vanilla EEGLab has trouble plotting
% the WSSP lab's driver monitoring data.

%labelSet = plotLabeledData(tTestingSet, tModel, tResults, 'srate', 250, 'includeClasses', {'Eye Blink'});

% I'm not sure how DETECT is handling the two sets having different numbers
% of channels.

%disp('Done.');

% Results

