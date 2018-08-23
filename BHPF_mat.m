function [tensor_hat,FactorMat,rmse] = BHPF_mat(original_mat,sparse_mat,varargin)
% Bayesian Hierarchical Poisson Factorization 

dim = size(sparse_mat);
d = length(dim);
% position = find(sparse_mat~=0);
pos = find(original_mat>0 & sparse_mat==0); 
% pos = find(sparse_tensor==0 );
% binary_tensor = zeros(dim);
% binary_tensor(position) = 1;

ip = inputParser;
ip.addParamValue('CP_rank',20,@isscalar);
ip.addParamValue('maxiter',200,@isscalar);
ip.parse(varargin{:});

r_CP = ip.Results.CP_rank;
maxiter = ip.Results.maxiter;

U = cell(d,1);
beta = cell(d,1);
for k = 1:d
    U{k} = rand(dim(k),r_CP);
    beta{k} = rand(dim(k),1);
end

alpha = 0.3;
a0 = 0.3;
b0 = 1;

% figure;
rmse = zeros(maxiter,1);
fprintf('\n------Bayesian Hierarchical Poisson Factorization------\n');
for iter = 1:maxiter
    % Generate X_ijr
    mat = kr(U{2},U{1});
    mat = mat./sum(mat,2);
    vec = sparse_mat(:);
    pos0 = find(vec>0);
    new_mat = zeros(dim(1)*dim(2),r_CP);
    new_mat(pos0,:) = mnrnd(vec(pos0),mat(pos0,:));
    X = zeros(dim(1), dim(2), r_CP);
    for r = 1:r_CP
        X(:,:,r) = reshape(new_mat(:,r),[dim(1),dim(2)]);
    end
    
    % Update hyper-parameter beta
    for k = 1:d
        for idx = 1:dim(k)
            beta{k}(idx) = gamrnd(a0+r_CP*alpha, 1./(a0./b0+sum(U{k}(idx,:))));
        end
    end
    
	% Update factor matrices U
    for i = 1:dim(1)
        for r = 1:r_CP
            X_id = sum(sum(X(i,:,r)));
            sum_factor = sum(U{2}(:,r));
            U{1}(i,r) = gamrnd(alpha+X_id, 1./(beta{1}(i)+sum_factor));
        end
    end
    for j = 1:dim(2)
        for r = 1:r_CP
            X_jd = sum(sum(X(:,j,r)));
            sum_factor = sum(U{1}(:,r));
            U{2}(j,r) = gamrnd(alpha+X_jd, 1./(beta{2}(j)+sum_factor));
        end
    end
	
    % Compute RMSE
	factor = cp_combination(U,dim);
	tensor_hat = round(factor);
	rmse(iter,1) = sqrt(sum((original_mat(pos)-tensor_hat(pos)).^2)./length(pos));
	
	% Print the results
    fprintf('#iteration = %g, #RMSE = %g  \n',iter,rmse(iter));
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    for k = 1:d
        subplot(1,d+3,k);imagesc(U{k});
    end
    subplot(1,d+3,d+1:d+3);plot(rmse(1:iter));
    ylim([4,10]);xlabel('iteration');ylabel('RMSE (km/h)');
    drawnow;
end
FactorMat = U;
