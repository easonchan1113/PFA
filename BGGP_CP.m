function [tensor_hat,FactorMat,rmse] = BGGP_CP(original_tensor,sparse_tensor,varargin)
% Bayesian CP Factorization for Count Data with BGGP assumption

dim = size(sparse_tensor);
d = length(dim);
position = find(sparse_tensor~=0);
pos = find(original_tensor>0 & sparse_tensor==0);
binary_tensor = zeros(dim);
binary_tensor(position) = 1;

ip = inputParser;
ip.addParamValue('CP_rank',20,@isscalar);
ip.addParamValue('maxiter',200,@isscalar);
ip.parse(varargin{:});

r_CP = ip.Results.CP_rank;
maxiter = ip.Results.maxiter;

U = cell(d,1);
r = cell(d,1);
p = cell(d,1);
for k = 1:d
    U{k} = 0.1*rand(dim(k),r_CP);
    r{k} = 0.1*rand(dim(k),1);
    p{k} = rand(dim(k),1);
end

a0 = 1e-6;
b0 = 1e6;
c0 = 1e-6;
d0 = 1e-6;

% figure;
rmse = zeros(maxiter,1);
fprintf('\n------Bayesian CP Factorization for Count Data with BGGP assumption------\n');
for iter = 1:maxiter
	% Update x_{nmk}
%     for i = 1:dim(1)
%         for j = 1:dim(2)
%             for k = 1:dim(3)
%                 if sparse_tensor(i,j,k) > 0
%                     xi = U{1}(i,:).*U{2}(j,:).*U{3}(k,:);
%                     xi = xi./sum(xi);
%                     X(i,j,k,:) = mnrnd(sparse_tensor(i,j,k),xi);
%                 end
%             end
%         end
%     end
    
    mat = kr(kr(U{3},U{2}),U{1});
    mat = mat./sum(mat,2);
% ten4 = zeros(dim(1),dim(2),dim(3),r_CP);
% for i = 1:r_CP
%     ten4(:,:,:,i) = reshape(mat(:,i),[dim(1),dim(2),dim(3)]);
% end
%     ten4 = reshape(mat,[dim(1),dim(2),dim(3),r_CP]);
    vec = sparse_tensor(:);
    pos0 = find(vec>0);
    new_mat = zeros(dim(1)*dim(2)*dim(3),r_CP);
%     for i = 1:length(pos0)
%         new_mat(pos0(i),:) = mnrnd(vec(pos0(i)),mat(pos0(i),:));
%     end
%     new_mat0 = new_mat;

    new_mat(pos0,:) = mnrnd(vec(pos0),mat(pos0,:));
%     new_mat0 = zeros(dim(1)*dim(2)*dim(3),r_CP);
%     new_mat0(pos0,:) = new_mat;
%     X = zeros(dim(1),dim(2),dim(3),r_CP);
    X = zeros(dim(1), dim(2), dim(3), r_CP);
    for i = 1:r_CP
        X(:,:,:,i) = reshape(new_mat(:,i),[dim(1),dim(2),dim(3)]);
    end
    
%     for i1 = 1:dim(1)
%         for i2 = 1:dim(2)
%             for i3 = 1:dim(3)
%                 if original_tensor(i1,i2,i3) > 0
%                     X(i1,i2,i3,:) = mnrnd(original_tensor(i1,i2,i3),reshape(ten4(i1,i2,i3,:),[1,r_CP]));
%                 end
%             end
%         end
%     end
        
	% Update factor matrices U^{(1)}
    for n = 1:dim(1)
        for k = 1:r_CP
            X_nk = sum(sum(X(n,:,:,k)));
            var1 = sum(sum(U{2}(:,k)*U{3}(:,k)'));
            U{1}(n,k) = gamrnd(r{1}(n,1)+X_nk, p{1}(n,1)./(1-p{1}(n,1)+p{1}(n,1)*var1));
        end
    end
    for n = 1:dim(2)
        for k = 1:r_CP
            X_nk = sum(sum(X(:,n,:,k)));
            var1 = sum(sum(U{1}(:,k)*U{3}(:,k)'));
            U{2}(n,k) = gamrnd(r{2}(n,1)+X_nk, p{2}(n,1)./(1-p{2}(n,1)+p{2}(n,1)*var1));
        end
    end
    for n = 1:dim(3)
        for k = 1:r_CP
            X_nk = sum(sum(X(:,:,n,k)));
            var1 = sum(sum(U{1}(:,k)*U{2}(:,k)'));
            U{3}(n,k) = gamrnd(r{3}(n,1)+X_nk, p{3}(n,1)./(1-p{3}(n,1)+p{3}(n,1)*var1));
        end
    end
	
	factor = cp_combination(U,dim);
	tensor_hat = round(factor);
	rmse(iter,1) = sqrt(sum((original_tensor(pos)-tensor_hat(pos)).^2)./length(pos));
	
	% Update hyper-parameter p
    for n = 1:dim(1)
        X_n = sum(sum(sum(X(n,:,:,:))));
%         N = length(find(X(n,:,:,:)>0));
        N = length(find(sparse_tensor(n,:,:))>0).*r_CP;
        p{1}(n,1) = betarnd(c0+X_n, d0+N*r{1}(n,1));
    end
    for n = 1:dim(2)
        X_n = sum(sum(sum(X(:,n,:,:))));
        N = length(find(sparse_tensor(:,n,:))>0).*r_CP;
        p{2}(n,1) = betarnd(c0+X_n, d0+N*r{2}(n,1));
    end
    for n = 1:dim(3)
        X_n = sum(sum(sum(X(:,:,n,:))));
        N = length(find(sparse_tensor(:,:,n))>0).*r_CP;
        p{3}(n,1) = betarnd(c0+X_n, d0+N*r{3}(n,1));
    end
	
	% Update hyper-parameter r
    for n = 1:dim(1)
        N = length(find(sparse_tensor(n,:,:))>0).*r_CP;
        L = 0;
        for x=X(n,:,:,:)
            l=0;
            if x>0
                for t=1:x
                    suc = r{1}(n,1)./(t-1+r{1}(n,1));
                    b_t = rand(1) < suc; % bernoulli random number
                    l = l + b_t;
                end
            end
            L = L+l;
        end
        r{1}(n,1) = gamrnd(a0+L, 1./(b0-N*log(1-p{1}(n,1))));
    end
    for n = 1:dim(2)
        N = length(find(sparse_tensor(:,n,:))>0).*r_CP;
        L = 0;
        for x=X(:,n,:,:)
            l=0;
            if x>0
                for t=1:x
                    suc = r{2}(n,1)./(t-1+r{2}(n,1));
                    b_t = rand(1) < suc; % bernoulli random number
                    l = l + b_t;
                end
            end
            L = L+l;
        end
        r{2}(n,1) = gamrnd(a0+L, 1./(b0-N*log(1-p{2}(n,1))));
    end
    for n = 1:dim(3)
        N = length(find(sparse_tensor(:,:,n))>0).*r_CP;
        L = 0;
        for x=X(:,:,n,:)
            l=0;
            if x>0
                for t=1:x
                    suc = r{3}(n,1)./(t-1+r{3}(n,1));
                    b_t = rand(1) < suc; % bernoulli random number
                    l = l + b_t;
                end
            end
            L = L+l;
        end
        r{3}(n,1) = gamrnd(a0+L, 1./(b0-N*log(1-p{3}(n,1))));
    end
	
	
	% Print the results
    fprintf('#iteration = %g, #RMSE = %g km/h \n',iter,rmse(iter));
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    for k = 1:d
        subplot(1,d+3,k);imagesc(U{k});
    end
    subplot(1,d+3,d+1:d+3);plot(rmse(1:iter));
    ylim([3.0,6.0]);xlabel('iteration');ylabel('RMSE (km/h)');
    drawnow;
end
FactorMat = U;