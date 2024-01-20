close all, clear all, clc

srate = 128;
Ts = 1/srate; % sampling rate of 128 Hz

%% take the class information
rootDataFolder = "C:\Users\Brhan\OneDrive\Belgeler\Ku Online\Fall23\Elec447\Project\data\BLSTM\DATA";

classFolders = dir(rootDataFolder);  % Get a list of class folders

%% iterate through class folders
trainData = [];
testData = [];

t_seg = 4; % length of each segments [in seconds]

for i = 3:length(classFolders)  % Skip first two entries (. and ..)
    currentClassFolder = classFolders(i);
    % Process data within this class folder

    fprintf('Class #%d: %s\n',i-3,currentClassFolder.name)

    % load data within class folders
    data = signalDatastore([currentClassFolder.folder, '\', currentClassFolder.name]);

    % split the data into test and training datasets
    [idxTrain, idxTest] = trainingPartitions(numel(data.Files),[0.5,0.5]);  % 50% training, 50% test

    sdsTrain = signalDatastore(data.Files(idxTrain));
    sdsTest  = signalDatastore(data.Files(idxTest));

    sdsTrain = cellfun(@transpose,sdsTrain.readall(),'UniformOutput',false);
    sdsTest  = cellfun(@transpose,sdsTest.readall(),'UniformOutput',false);

    % apply necessary operations [HERE]
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

    


    % prepare the training and test data
    labels = (repmat(currentClassFolder.name,size(sdsTrain)));
    currentTrainData = [sdsTrain, cellstr(labels)];  % Read training data and labels    
    labels = (repmat(currentClassFolder.name,size(sdsTest)));
    currentTestData  = [sdsTest, cellstr(labels)];    % Read test data

    % combine with the previous classes
    trainData = [trainData;currentTrainData];
    testData  = [testData;currentTestData];


end

%% Window the data into smaller segments



