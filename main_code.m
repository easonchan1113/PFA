%% initial
clear;
low_rank = 15; % modify the low rank here
missing_rate = 0.1; % modify the missing rate here

%% missing setting
load ml_mat;
pos = find(ml_mat>0);
pos0 = randsample(pos, round(length(pos)*missing_rate));
ml_mat_sparse = ml_mat;
ml_mat_sparse(pos0) = 0;

%% running code
% [tensor_hat, FactorMat, accuracy] = BGGP(original_tensor, sparse_tensor, 'CP_rank',low_rank,'maxiter',200);
% [tensor_hat, FactorMat, accuracy] = GaP_v1(original_tensor, sparse_tensor, 'CP_rank',low_rank,'maxiter',200);
% [tensor_hat, FactorMat, accuracy] = GaP_v2(original_tensor, sparse_tensor, 'CP_rank',low_rank,'maxiter',1000);
% [tensor_hat, FactorMat, accuracy] = BPTF(original_tensor, sparse_tensor, 'CP_rank',low_rank,'maxiter',500);
[tensor_hat1, FactorMat1, accuracy1] = BHPF_mat(ml_mat, ml_mat_sparse, 'CP_rank',low_rank,'maxiter',200);
