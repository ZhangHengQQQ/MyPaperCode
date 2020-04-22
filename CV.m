clear
root=cd;
load coding
nSubj=length(lab);
kfoldout=nSubj;
cv = cvpartition(nSubj,'k',kfoldout); % for outer CV
cov_type='full';
O = 2;
T = 80;
M = 3;
Q = 7;
prior0 = normalise(rand(Q,1));
transmat0 = mk_stochastic(rand(Q,Q));
for f=1:cv.NumTestSets
    train_data=encode(training(cv,f));
    train_lab=lab(training(cv,f));
    test_data=encode(test(cv,f));
    test_lab=lab(test(cv,f));
    MCIdata=train_data(logical(train_lab-1));
    for i=1:length(MCIdata)
        trainMCIdata(:,:,i)=MCIdata{i}';
    end
    NCdata=train_data(logical(train_lab+1));
    for i=1:length(NCdata)
        trainNCdata(:,:,i)=NCdata{i}';
    end
    
    %% ÑµÁ·MCI
    MCInex=size(trainMCIdata,3);
    [mu0, Sigma0] = mixgauss_init(Q*M, reshape(trainMCIdata, [O T*MCInex]), cov_type);
    mu0 = reshape(mu0, [O Q M]);
    Sigma0 = reshape(Sigma0, [O O Q M]);
    mixmat0 = mk_stochastic(rand(Q,M));
    [LL, MCIprior, MCItransmat, MCImu, MCISigma, MCImixmat] = ...
        mhmm_em(trainMCIdata, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 1000);
    %% ÑµÁ·NC
    NCnex=size(trainNCdata,3);
    [mu0, Sigma0] = mixgauss_init(Q*M, reshape(trainNCdata, [O T*NCnex]), cov_type);
    mu0 = reshape(mu0, [O Q M]);
    Sigma0 = reshape(Sigma0, [O O Q M]);
    mixmat0 = mk_stochastic(rand(Q,M));
    [LL, NCprior, NCtransmat, NCmu, NCSigma, NCmixmat] = ...
        mhmm_em(trainNCdata, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 1000);
    %% ²âÊÔ
    MCIloglik = mhmm_logprob(test_data{1}', MCIprior, MCItransmat, MCImu, MCISigma, MCImixmat);
    NCloglik = mhmm_logprob(test_data{1}', NCprior, NCtransmat, NCmu, NCSigma, NCmixmat);
    %     if MCIloglik>0||NCloglik>0
    %         pause;
    %     end
    test_result=NCloglik>MCIloglik;
    true_result=logical(test_lab+1);
    trr(f)=true_result;
    ter(f)=test_result;
    result(f)=(test_result==true_result);
    
end
sum(result)/length(result)