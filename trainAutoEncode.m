clear
load('fMRI80.mat')
fMRIdata = fMRImciNc;
numSubjects = length(fMRIdata);
[numScans, numROIs] = size(fMRIdata{1});
trainSet = cell2mat( fMRIdata );
trainSet = reshape( trainSet, [numScans, numROIs,numSubjects]);
trainSet = reshape( trainSet, [numScans * numSubjects, numROIs]);
trainSet = zscore(trainSet);
save processedData trainSet lab subIdx
numLayerUnits = [200, 150, 10, 2];
randIdex = randperm(size(trainSet, 1));
rbmTrainSet = trainSet(randIdex, :);
numHiddens = 100;
learningRate = 0.01;
maxepoch = 30;
numRBMCD = 1;
maxIter = 100;
miniBatchSize = 10;
% sparse RBM
targetSparsity = 0.05;
sparsityControl = 0.1;
L2Regularizer = 0.001; %0.0001
dbnParams = cell(1, length(numLayerUnits));
rbmTrainParams = cell(1, length(numLayerUnits));
for i=1:length(numLayerUnits)
    rbmTrainParams{i} = RbmSetTrainParams([], numLayerUnits(i), ...
        learningRate,L2Regularizer, targetSparsity, sparsityControl, ...
        numRBMCD, maxIter, miniBatchSize, cd );
    if i==1
        rbmTrainParams{i}.visIsBinary = 0;   % real-valed
        rbmTrainParams{i}.hidIsBinary = 1;
        rbmTrainParams{i}.kCD = 1;
        rbmTrainParams{i}.numVis = numROIs;
    elseif i==length(numLayerUnits)
        rbmTrainParams{i}.visIsBinary = 1;   % bianry
        rbmTrainParams{i}.hidIsBinary = 0;
        rbmTrainParams{i}.numVis = numLayerUnits(i-1);
    else
        rbmTrainParams{i}.visIsBinary = 1;   % bianry
        rbmTrainParams{i}.hidIsBinary = 1;
        rbmTrainParams{i}.numVis = numLayerUnits(i-1);
    end
    dbnParams{i} = RbmTrain(rbmTrainSet, rbmTrainParams{i});
    if rbmTrainParams{1}.hidIsBinary == 1
        rbmTrainSet = sigmoid(rbmTrainSet * dbnParams{i}.visHid + repmat(dbnParams{i}.hidBias, size(rbmTrainSet, 1), 1));
    else
        rbmTrainSet = rbmTrainSet * dbnParams{l}.visHid + repmat(dbnParams{l}.hidBias, size(rbmTrainSet, 1), 1);
    end
end
eval( ['save ', 'preTrainDBN', ' dbnParams']);
totalSamples = size(trainSet, 1);
numTrainSamples = ceil( totalSamples * 0.9 / 100 ) * 100;
randIdex = randperm( totalSamples );
trainData = trainSet(randIdex(1:numTrainSamples), :);
validData = trainSet(randIdex((numTrainSamples + 1):end), :);
daeParams = Backprob(trainData, validData, dbnParams, 100 );
DAE=dbnParams;
for i=1:length(numLayerUnits)
    DAE{i}.visHid = daeParams{i}(1:end-1, :);
    DAE{i}.hidBias = daeParams{i}(end, :);
end
save DAE DAE
save rbmTrainParams rbmTrainParams