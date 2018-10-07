function [mat_hat,FactorMat,rmse] = BHPF_VB(original_mat,sparse_mat,varargin)
% Bayesian Hierarchical Poisson Factorization 
% Implement of "Scalable Recommendation with Hierarchical Poisson Factorization"

dim = size(sparse_mat);
D = length(dim);
position = sparse_mat~=0;
pos = find(original_mat>0 & sparse_mat==0); 
% pos = find(sparse_tensor==0 );
binary_tensor = zeros(dim);
binary_tensor(position) = 1;

ip = inputParser;
ip.addParameter('CP_rank',20,@isscalar);
ip.addParameter('maxiters',200,@isscalar);
ip.parse(varargin{:});

K = ip.Results.CP_rank;
maxiters = ip.Results.maxiters;

a = 0.3;
a0 = 0.3;
b0 = 1;

U = cell(D,1);
beta = cell(D,1);
a0_new = a0+K*a;
b0_new = cell(D,1);
a_new = cell(D,1);
b_new = cell(D,1);
E_lnU = cell(D,1);
for d = 1:D
    U{d} = rand(dim(d),K);
    beta{d} = rand(dim(d),1);
    b0_new{d} = zeros(dim(d),1);
    a_new{d} = zeros(dim(d),K);
    b_new{d} = zeros(dim(d),K);
end

LB = 0;

% figure;
rmse = zeros(maxiters,1);
fprintf('\n------Bayesian Hierarchical Poisson Factorization------\n');
fprintf('\n-----------Using Variational Bayes Inference-----------\n');
for iter = 1:maxiters
    % Update auxiliary variable z_ijk
    for d = 1:D
        E_lnU{d} = psi(a_new{d})-safelog(b_new{d});
    end
    Z = zeros(dim(1), dim(2), K);
    [rows, cols, vs] = find(sparse_mat);
    for idx = 1:length(vs)
        phi_ij = exp(E_lnU{1}(rows(idx),:) + E_lnU{2}(cols(idx),:));
        phi_ij = phi_ij./sum(phi_ij);
        Z(rows(idx), cols(idx), :) = vs(idx)*phi_ij;
    end
    
    % Update hyper-parameter beta
    for d = 1:D
        for idx = 1:dim(d)
            b0_new{d}(idx) = a0./b0+sum(U{d}(idx,:));
            beta{d}(idx) = a0_new./b0_new{d}(idx);
        end
    end
    
	% Update factor matrices U
    for i = 1:dim(1)
        for k = 1:K
            z_ik = sum(Z(i,:,k));
            a_new{1}(i,k) = a + z_ik;
            sum_factor = sum(U{2}(:,k));
            b_new{1}(i,k) = beta{1}(i) + sum_factor;
            U{1}(i,k) = a_new{1}(i,k)./b_new{1}(i,k);
        end
    end
    for j = 1:dim(2)
        for k = 1:K
            z_jk = sum(Z(:,j,k));
            a_new{2}(j,k) = a + z_jk;
            sum_factor = sum(U{1}(:,k));
            b_new{2}(j,k) = beta{2}(j) + sum_factor;
            U{2}(j,k) = a_new{2}(j,k)./b_new{2}(j,k);
        end
    end
	
    % Evaluate lower bound
    
    % Compute RMSE
	factor = U{1}*U{2}';
	mat_hat = round(factor);
	rmse(iter,1) = sqrt(sum((original_mat(pos)-mat_hat(pos)).^2)./length(pos));
	
	% Visualizing
    fprintf('#iteration = %g, #RMSE = %g  \n',iter,rmse(iter));
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    for d = 1:D
        subplot(1,D+3,d);imagesc(U{d});
    end
    subplot(1,D+3,D+1:D+3);plot(rmse(1:iter));
    ylim([4,10]);xlabel('iteration');ylabel('RMSE (km/h)');
    drawnow;
end
FactorMat = U;
