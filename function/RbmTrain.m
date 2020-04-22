function [wRbm] = RbmTrain(trainSet,trainParams)
fprintf('numVisible:%d----numHidden:%d\n', trainParams.numVis, trainParams.numHid);
initialMomentum = 0.5;
finalMomentum = 0.9;
wRbm.visHid = 0.02*randn(trainParams.numVis, trainParams.numHid);
wRbm.visBias = zeros(1, trainParams.numVis);
wRbm.hidBias = zeros(1, trainParams.numHid);
shared.visHidInc = zeros(size(wRbm.visHid));
shared.hidBiasInc = zeros(size(wRbm.hidBias));
shared.visBiasInc = zeros(size(wRbm.visBias));
epsilon = trainParams.epsilon;
shared.runningAvgProb = [];
errorHistory = zeros(trainParams.maxIter,1);
sparsityHistory = zeros(trainParams.maxIter,1);
N = size(trainSet,1);
numIter = min(floor(N/trainParams.batchSize), 1000); % changed ceil -> floor
for epoch = 1:trainParams.maxIter
    reconErr = zeros(numIter, 1);
    sparsityErr = zeros(numIter, 1);
    if epoch < 5
        momentum = initialMomentum;
    else
        momentum = finalMomentum;
    end
    randIdex = randperm(N);
    for iter = 1:numIter
        batchIdex = randIdex((iter - 1) * trainParams.batchSize ...
            + 1 : iter * trainParams.batchSize);
        trainData = trainSet(batchIdex, :);
        visBiasMat = repmat(wRbm.visBias, trainParams.batchSize, 1);
        hidBiasMat = repmat(wRbm.hidBias, trainParams.batchSize, 1);
        if trainParams.hidIsBinary == 1
            posHidProb = sigmoid(trainData * wRbm.visHid + hidBiasMat);
            posHidStates = rand(size(posHidProb)) < posHidProb;
        else
            posHidProb = trainData * wRbm.visHid + hidBiasMat;
            posHidStates = rand(size(posHidProb)) < posHidProb;
        end
        negHidStates = posHidStates;
        for k=1:trainParams.kCD
            if trainParams.visIsBinary == 1
                negData = sigmoid(negHidStates * wRbm.visHid' + visBiasMat);
            else
                negData = negHidStates * wRbm.visHid' + visBiasMat;
            end
            if trainParams.hidIsBinary == 1
                negHidProb = sigmoid(negData * wRbm.visHid + hidBiasMat);
                negHidStates = rand(size(negHidProb)) < negHidProb;
            else
                negHidProb = negData * wRbm.visHid + hidBiasMat;
                negHidStates = rand(size(negHidProb)) + negHidProb;
            end
            
        end
        if trainParams.visIsBinary == 1
            recon = sigmoid(posHidProb * wRbm.visHid' + visBiasMat);
        else
            recon = posHidProb * wRbm.visHid' + visBiasMat;
        end
        reconErr(iter) = norm(recon - trainData, 'fro') / trainParams.batchSize;
        sparsityErr(iter) = mean(posHidProb(:));
        
        diffVisHin = (trainData' * posHidProb - negData' * negHidProb) ...
            / trainParams.batchSize;
        diffHinBias = mean(posHidProb - negHidProb);
        diffVisBias = mean(trainData - negData);
        if isempty(shared.runningAvgProb)
            shared.runningAvgProb = mean(posHidProb);
        else
            shared.runningAvgProb = 0.9 * shared.runningAvgProb + 0.1 * mean(posHidProb);
        end
        diffHinBiasSparsity = trainParams.pLambda * (trainParams.pBias - shared.runningAvgProb);
        
        % update parameters
        shared.visHidInc = momentum * shared.visHidInc + epsilon ...
            * (diffVisHin - trainParams.L2Reg * wRbm.visHid);
        shared.hidBiasInc = momentum * shared.hidBiasInc + epsilon ...
            * (diffHinBias + diffHinBiasSparsity);
        shared.visBiasInc = momentum * shared.visBiasInc + epsilon * diffVisBias;
        wRbm.visHid = wRbm.visHid + shared.visHidInc;
        wRbm.hidBias = wRbm.hidBias + shared.hidBiasInc;
        wRbm.visBias = wRbm.visBias + shared.visBiasInc;
    end
    errorHistory(epoch) = mean(reconErr);
    sparsityHistory(epoch) = mean(sparsityErr);
    fprintf('epoch %d:\t error=%g,\t sparsity=%g\n', epoch, errorHistory(epoch), sparsityHistory(epoch));
end
saveFileName=['numVisible',num2str(trainParams.numVis),'_numHidden',num2str(trainParams.numHid),'.mat'];
eval( ['save ', saveFileName, ' wRbm']);
end

