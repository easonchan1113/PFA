function [tensor_hat,FactorMat,rmse] = BHPF(original_tensor,sparse_tensor,varargin)
% Bayesian Hierarchical Poisson Factorization 

dim = size(sparse_tensor);
d = length(dim);
position = find(sparse_tensor~=0);
pos = find(original_tensor>0 & sparse_tensor==0); 
% pos = find(sparse_tensor==0 );
binary_tensor = zeros(dim);
binary_tensor(position) = 1;

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

alpha = 5;
a0 = 0.5;
b0 = 1;

% figure;
rmse = zeros(maxiter,1);
fprintf('\n------Bayesian Poisson Tensor Factorization------\n');
for iter = 1:maxiter
    % Generate X_ijkr
    mat = kr(kr(U{3},U{2}),U{1});
    mat = mat./sum(mat,2);
    vec = sparse_tensor(:);
    pos0 = find(vec>0);
    new_mat = zeros(dim(1)*dim(2)*dim(3),r_CP);
    new_mat(pos0,:) = mnrnd(vec(pos0),mat(pos0,:));
    X = zeros(dim(1), dim(2), dim(3), r_CP);
    for r = 1:r_CP
        X(:,:,:,r) = reshape(new_mat(:,r),[dim(1),dim(2),dim(3)]);
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
            X_id = sum(sum(X(i,:,:,r)));
            sum_factor = sum(sum(U{2}(:,r)*U{3}(:,r)'));
            U{1}(i,r) = gamrnd(alpha+X_id, 1./(beta{1}(i)+sum_factor));
        end
    end
    for j = 1:dim(2)
        for r = 1:r_CP
            X_jd = sum(sum(X(:,j,:,r)));
            sum_factor = sum(sum(U{1}(:,r)*U{3}(:,r)'));
            U{2}(j,r) = gamrnd(alpha+X_jd, 1./(beta{2}(j)+sum_factor));
        end
    end
    for k = 1:dim(3)
        for r = 1:r_CP
            X_kd = sum(sum(X(:,:,k,r)));
            sum_factor = sum(sum(U{1}(:,r)*U{2}(:,r)'));
            U{3}(k,r) = gamrnd(alpha+X_kd, 1./(beta{3}(k)+sum_factor));
        end
    end
	
    % Compute RMSE
	factor = cp_combination(U,dim);
	tensor_hat = round(factor);
	rmse(iter,1) = sqrt(sum((original_tensor(pos)-tensor_hat(pos)).^2)./length(pos));
	
	% Print the results
    fprintf('#iteration = %g, #RMSE = %g  \n',iter,rmse(iter));
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    for k = 1:d
        subplot(1,d+3,k);imagesc(U{k});
    end
    subplot(1,d+3,d+1:d+3);plot(rmse(1:iter));
    ylim([6,10.5]);xlabel('iteration');ylabel('RMSE (km/h)');
    drawnow;
end
FactorMat = U;