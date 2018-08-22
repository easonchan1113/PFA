function [X_hat] = HaLRTC(original_tensor,sparse_tensor,maxiter)
dim = size(original_tensor);
n1 = dim(1);n2 = dim(2); n3 = dim(3);
pos_obs = find(sparse_tensor~=0); % index set of observed entries
S = zeros(dim);
S(pos_obs) = 1;
pos_unobs = find(sparse_tensor==0 & original_tensor~=0 ); % index set of missing entries
alpha = ones(1,3)./3;
rho = 10^(-3);
X_hat = sparse_tensor;
Y1 = zeros(dim); Y2 = Y1; Y3=Y1; % additive tensor
rmse = zeros(maxiter,1);
for iter = 1:maxiter
    [u1,s1,v1] = svds(ten2mat(X_hat,dim,1)+ten2mat(Y1,dim,1)/rho, n1);
    B1 = mat2ten(u1*diag(max(diag(s1)-alpha(1)/rho,0))*v1',dim,1); % update tensor B1
    [u2,s2,v2] = svds(ten2mat(X_hat,dim,2)+ten2mat(Y2,dim,2)/rho, n2);
    B2 = mat2ten(u2*diag(max(diag(s2)-alpha(2)/rho,0))*v2',dim,2); % update tensor B2
    [u3,s3,v3] = svds(ten2mat(X_hat,dim,3)+ten2mat(Y3,dim,3)/rho, n3);
    B3 = mat2ten(u3*diag(max(diag(s3)-alpha(3)/rho,0))*v3',dim,3); % update tensor B3
    X_hat = (1-S).*(B1+B2+B3-(Y1+Y2+Y3)/rho)/3+S.*sparse_tensor; % update the estimated tensor
    Y1 = Y1-rho*(B1-X_hat);
    Y2 = Y2-rho*(B2-X_hat);
    Y3 = Y3-rho*(B3-X_hat);
    rmse(iter,1) = sqrt(sum((original_tensor(pos_unobs)-X_hat(pos_unobs)).^2)./length(pos_unobs));
    fprintf('iteration = %g, RMSE = %g km/h.\n',iter,rmse(iter));
end