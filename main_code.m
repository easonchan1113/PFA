%% initial
clear;
n1 = 214;
n2 = 61;
low_rank = 20; % modify the low rank here

%% missing setting
missing_rate = 0.2; % modify the missing rate here
ratio = 0.1*missing_rate;
load tensor;
load random_tensor;
[original_tensor,sparse_tensor] = missing_setting(tensor,random_tensor,'order',3,'missing_scenario','element','missing_rate',ratio);
original_tensor1 = original_tensor(1:n1,1:n2,:);
original_tensor = round(original_tensor1); % round the speed value as an integer
sparse_tensor1 = sparse_tensor(1:n1,1:n2,:);
sparse_tensor = round(sparse_tensor1);

%% running code
% [tensor_hat, FactorMat, accuracy] = BGGP_CP(original_tensor, sparse_tensor, 'CP_rank',low_rank,'maxiter',200);
% [tensor_hat, FactorMat, accuracy] = GaP_CP_v1(original_tensor, sparse_tensor, 'CP_rank',low_rank,'maxiter',200);
% [tensor_hat1, FactorMat1, accuracy1] = GaP_CP_v2(original_tensor, sparse_tensor, 'CP_rank',low_rank,'maxiter',200);
% [tensor_hat, FactorMat, accuracy] = BPTF(original_tensor, sparse_tensor, 'CP_rank',low_rank,'maxiter',200);
[tensor_hat1, FactorMat1, accuracy1] = BHPF(original_tensor, sparse_tensor, 'CP_rank',low_rank,'maxiter',200);
