close all, clear all, clc

%% take the class information
rootDataFolder = "C:\Users\Brhan\OneDrive\Belgeler\Ku Online\Fall23\Elec447\Project\data\BLSTM\DATA";
 
classFolders = dir(rootDataFolder);  % Get a list of class folders

%% set a minibatch size
miniBatchSize = 100;

%% define the lstm network architecture
inputSize = 95; % num of channels
numHiddenUnits = 200;
numClasses = 2;

layers = [ ...
    featureInputLayer(inputSize)
    fullyConnectedLayer(numHiddenUnits)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

%% define the training options
options = trainingOptions("adam", ...
    ExecutionEnvironment="gpu", ...
    InitialLearnRate=1e-3, ...
    GradientThreshold=1,...
    MaxEpochs=30, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Verbose=0, ...
    Plots="training-progress");

%% iterate through class folders

% trainData = [];
% testData = [];
% for i = 3:length(classFolders)  % Skip first two entries (. and ..)
%     currentClassFolder = classFolders(i);
%     % Process data within this class folder
% 
%     fprintf('Class #%d: %s\n',i-3,currentClassFolder.name)
% 
%     % load data within class folders
%     data = signalDatastore([currentClassFolder.folder, '\', currentClassFolder.name]);
% 
%     % split the data into test and training datasets
%     [idxTrain, idxTest] = trainingPartitions(numel(data.Files),[0.8,0.2]);  % 50% training, 50% test
% 
%     sdsTrain = signalDatastore(data.Files(idxTrain));
%     sdsTest  = signalDatastore(data.Files(idxTest));
% 
%     % transpose the output before training the neural network
%     sdsTrain = cellfun(@transpose,sdsTrain.readall(),'UniformOutput',false);
%     sdsTest  = cellfun(@transpose,sdsTest.readall(),'UniformOutput',false);
% 
%     % apply necessary operations [HERE]
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%     sdsTrain = cellfun(@clean2,sdsTrain,'UniformOutput',false);
%     sdsTest  = cellfun(@clean2,sdsTest,'UniformOutput',false);
% 
%     extractedCells = {};
%     for k=1:numel(sdsTrain)
%         extractedCells = [extractedCells;sdsTrain{k}];
%     end
%     sdsTrain = extractedCells;
% 
%     extractedCells = {};
%     for k=1:numel(sdsTest)
%         extractedCells = [extractedCells;sdsTest{k}];
%     end
%     sdsTest = extractedCells;
% 
%     % prepare the training and test data
%     labels = (repmat(currentClassFolder.name,size(sdsTrain)));
%     currentTrainData = [sdsTrain, cellstr(labels)];  % Read training data and labels    
%     labels = (repmat(currentClassFolder.name,size(sdsTest)));
%     currentTestData  = [sdsTest, cellstr(labels)];    % Read test data
% 
%     % combine with the previous classes
%     trainData = [trainData;currentTrainData];
%     testData  = [testData;currentTestData];
% 
% end
% 
% trainData = shuffle(trainData);
% 
% save('dataset_wo_jaded_clean.mat','trainData','testData','-mat')

load('dataset_jaded_clean.mat');
%% prepare the data
[N,~] = size(trainData);
XTrain = trainData(1:N)';
YTrain =  categorical( trainData(N+1:2*N)' );

[N,~] = size(testData);
XTest = testData(1:N)';
YTest = categorical( testData(N+1:2*N)' );

XTrain = cell2mat(cellfun(@(x) reshape(x, [], 95), XTrain, 'UniformOutput', false));
XTest  = cell2mat(cellfun(@(x) reshape(x, [], 95), XTest, 'UniformOutput', false));


% data = [trainData;testData];
% 
% N = numel(data);
% 
% A = data(1:N/2);
% B = categorical(data(N/2+1:N)');
% 
% % load('train_test_data.mat')
% N = numel(A);
% 
% [idxTrain, idxTest] = trainingPartitions(N,[0.8,0.2]);  % 50% training, 50% test
% 
% XTrain = A(idxTrain);
% YTrain = B(idxTrain);
% 
% XTest = A(idxTest);
% YTest = B(idxTest);

% %% short by the sequence lengths
% 
% numObservations = numel(XTrain);
% for i=1:numObservations
%     sequence = XTrain{i};
%     sequenceLengths(i) = size(sequence,2);
% end
% 
% [sequenceLengths,idx] = sort(sequenceLengths);
% XTrain = XTrain(idx);
% YTrain = YTrain(idx);
% 
% figure
% bar(sequenceLengths)
% xlabel("Sequence")
% ylabel("Length")
% title("Sorted Data")
% close all
%% train the network
net = trainNetwork(XTrain,YTrain,layers,options);

%% classify the test data
YPred = classify(net,XTest, ...
    MiniBatchSize=miniBatchSize, ...
    SequenceLength="longest");

% confusion chart
figure
confusionchart(YTest,YPred)

% output the classification performance metrics
[acc,sens,prec,f1s] = calculateMetrics(YTest,YPred)