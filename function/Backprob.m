function [trainedMLP] = Backprob(trainData, validData, dbnParams, maxepoch )
batchdata = reshape( trainData, [size(trainData,2), gcd(size(trainData,1), size(trainData,1)/100), 100] );
batchdata = permute( batchdata,[2, 1, 3] );
N = size(batchdata, 1);
w1 = [dbnParams{1}.visHid; dbnParams{1}.hidBias];
w2 = [dbnParams{2}.visHid; dbnParams{2}.hidBias];
w3 = [dbnParams{3}.visHid; dbnParams{3}.hidBias];
w4 = [dbnParams{4}.visHid; dbnParams{4}.hidBias];
w5 = [dbnParams{4}.visHid'; dbnParams{4}.visBias];
w6 = [dbnParams{3}.visHid'; dbnParams{3}.visBias];
w7 = [dbnParams{2}.visHid'; dbnParams{2}.visBias];
w8 = [dbnParams{1}.visHid'; dbnParams{1}.visBias];

l1 = size(w1, 1) - 1;
l2 = size(w2, 1) - 1;
l3 = size(w3, 1) - 1;
l4 = size(w4, 1) - 1;
l5 = size(w5, 1) - 1;
l6 = size(w6, 1) - 1;
l7 = size(w7, 1) - 1;
l8 = size(w8, 1) - 1;
l9 = l1;
trainErr=[];
for epoch = 1:maxepoch
    
    %%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err = 0;
    [numcases, numdims, numbatches] = size(batchdata);
    N = numcases;
    for batch = 1:numbatches
        data = [batchdata(:, :, batch)];
        data = [data, ones(N, 1)];
        w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
        w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
        w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
        w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
        w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
        w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N,1)];
        w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N,1)];
        dataout = 1./(1 + exp(-w7probs*w8));
        err = err +  1/N*sum(sum( (data(:,1:end-1)-dataout).^2 ));
    end
    trainErr(epoch) = err / numbatches;
    fprintf( 2, 'train error: %f\n', trainErr(epoch) );
    
    %%%%%%%%%%%%%%%%%%%% COMPUTE TEST RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N = size(validData, 1);
    err=0;
    data = validData;
    data = [data ones(N,1)];
    w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
    w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
    w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
    w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
    w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N,1)];
    w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N,1)];
    dataout = 1./(1 + exp(-w7probs*w8));
    err = err +  1/N*sum(sum( (data(:,1:end-1)-dataout).^2 ));
    testErr(epoch)=err;%/testnumbatches;
    fprintf(1,'Before epoch %d Train squared error: %6.3f Test squared error: %6.3f \t \t \n',epoch,trainErr(epoch),testErr(epoch));
    
    %%%%%%%%%%%%%% END OF COMPUTING TEST RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if epoch>2 && (testErr(epoch-1)-testErr(epoch) < eps)
        fprintf( 2, '\t Converged: %f\n', testErr(epoch-1)-testErr(epoch) );
        break;
    end
    
    tt=0;
    fprintf(1,'epoch %d: ', epoch);
    for batch = 1:numbatches/10
        fprintf( 1, '.' );
        %             fprintf(1,'epoch %d batch %d\r',epoch,batch);
        
        %%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tt=tt+1;
        data=[];
        for kk=1:10
            data=[data; batchdata(:,:,(tt-1)*10+kk)];
        end
        
        %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        max_iter = 3;
        VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)']';
        Dim = [l1; l2; l3; l4; l5; l6; l7; l8; l9];
        
        [X, fX] = minimize(VV, 'CG_MNIST', max_iter, Dim, data);
        
        w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
        xxx = (l1+1)*l2;
        w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
        xxx = xxx+(l2+1)*l3;
        w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
        xxx = xxx+(l3+1)*l4;
        w4 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
        xxx = xxx+(l4+1)*l5;
        w5 = reshape(X(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
        xxx = xxx+(l5+1)*l6;
        w6 = reshape(X(xxx+1:xxx+(l6+1)*l7),l6+1,l7);
        xxx = xxx+(l6+1)*l7;
        w7 = reshape(X(xxx+1:xxx+(l7+1)*l8),l7+1,l8);
        xxx = xxx+(l7+1)*l8;
        w8 = reshape(X(xxx+1:xxx+(l8+1)*l9),l8+1,l9);
        
        %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    trainedMLP{1} = w1;
    trainedMLP{2} = w2;
    trainedMLP{3} = w3;
    trainedMLP{4} = w4;
    trainedMLP{5} = w5;
    trainedMLP{6} = w6;
    trainedMLP{7} = w7;
    trainedMLP{8} = w8;
    
    fprintf( 1, '\n' );
    
    %     save mnist_weights w1 w2 w3 w4 w5 w6 w7 w8
    %     save mnist_error test_err train_err;
end
end

