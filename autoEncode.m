clear
load DAE
load processedData
load rbmTrainParams

for i=1:length(DAE)
    if rbmTrainParams{i}.hidIsBinary == 1
        trainSet = sigmoid(trainSet * DAE{i}.visHid + repmat(DAE{i}.hidBias, size(trainSet, 1), 1));
    else
        trainSet = trainSet * DAE{i}.visHid + repmat(DAE{i}.hidBias, size(trainSet, 1), 1);
    end
end
numScan = size(trainSet,1)/length(lab);

for i=1:length(lab)
    encode{i} = trainSet((i - 1) * numScan + 1:i * numScan,:);
end
save coding encode lab subIdx