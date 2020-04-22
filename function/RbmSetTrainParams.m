function params = RbmSetTrainParams(dataset,numhid,epsilon,l2reg,pbias,plambda,kCD,maxiter,batchsize,savePath)
params.dataSet = dataset;
params.visIsBinary = 1;          % added by HIS (Nov. 25, 2014)
params.hidIsBinary = 1;          % added by HIS (Nov. 25, 2014)
params.numHid = numhid;
params.epsilon = epsilon;
params.epsDecay = 0.01;
params.L2Reg = l2reg;
params.L1Reg = 0;
params.pBias = pbias;
params.pLambda = plambda;
params.kCD = kCD;
params.maxIter = maxiter;
params.batchSize = batchsize;
params.savePath = savePath;

end

