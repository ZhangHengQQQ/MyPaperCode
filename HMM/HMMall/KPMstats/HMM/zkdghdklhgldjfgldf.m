clear
O = 2;
T = 50;
nex = 50;
cov_type = 'full';
data = randn(O,T,nex);

M = 2;
Q = 2;
left_right = 0;

prior0 = normalise(rand(Q,1));
transmat0 = mk_stochastic(rand(Q,Q));

[mu0, Sigma0] = mixgauss_init(Q*M, reshape(data, [O T*nex]), cov_type);
mu0 = reshape(mu0, [O Q M]);
Sigma0 = reshape(Sigma0, [O O Q M]);
mixmat0 = mk_stochastic(rand(Q,M));

[LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
    mhmm_em(data, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 2);