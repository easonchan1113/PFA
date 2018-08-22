function [original_tensor,sparse_tensor] = missing_setting(original_tensor,random_tensor,varargin)

dim = size(original_tensor); % tensor size
n1 = dim(1);
n2 = dim(2);
n3 = dim(3);

ip = inputParser;
ip.addParamValue('order',3,@isscalar);
ip.addParamValue('missing_scenario','element',...
    @(x)ismember(x,{'element','fiber','slice'}));
ip.addParamValue('missing_rate',0.20,@isscalar);
ip.parse(varargin{:});

d = ip.Results.order;
missing_scenario = ip.Results.missing_scenario;
missing_rate = ip.Results.missing_rate;

if strcmp(missing_scenario,'element')
%     load random_tensor;
    rand_ten = random_tensor;
    if d == 3
        bin_tensor = round(rand_ten+0.5-missing_rate);
        sparse_tensor = original_tensor.*bin_tensor;
    elseif d == 4
        tensor = zeros(n1,9,7,n3);
        bin_tensor = zeros(n1,9,7,n3);
        for i1 = 1:n1
            for i2 = 1:n2
                tensor(i1,ceil(i2/7),i2-7*(ceil(i2/7)-1),:) = ...
                    reshape(original_tensor(i1,i2,:),[n3,1]);
                bin_tensor(i1,ceil(i2/7),i2-7*(ceil(i2/7)-1),:) = ...
                    reshape(round(rand_ten(i1,i2,:)+0.5-missing_rate),[n3,1]);
            end
        end
		sparse_tensor = tensor.*bin_tensor;
		original_tensor = tensor;
    end
elseif strcmp(missing_scenario,'fiber')
    load rand_ten_fiber;
    if d == 3
        bin_tensor=zeros(n1,n2,n3);
        for i1 = 1:n1
            for i2 = 1:n2
                bin_tensor(i1,i2,:)=round(rand_ten_fiber(i1,i2)+0.5-missing_rate);
            end
        end
        sparse_tensor = original_tensor.*bin_tensor;
    elseif d == 4
        tensor = zeros(n1,9,7,n3);
        bin_tensor = zeros(n1,9,7,n3);
        for i1 = 1:n1
            for i2 = 1:n2
                tensor(i1,ceil(i2/7),i2-7*(ceil(i2/7)-1),:) = ...
                    reshape(original_tensor(i1,i2,:),[n3,1]);
                for i3 = 1:n3
                    bin_tensor(i1,ceil(i2/7),i2-7*(ceil(i2/7)-1),:) = ...
                        round(rand_ten_fiber(i1,i2)+0.5-missing_rate);
                end
            end
        end
        sparse_tensor = tensor.*bin_tensor;
        original_tensor = tensor;
    end
elseif strcmp(missing_scenario,'slice')
        load rand_ten_slice;
        if d == 3
            bin_tensor = zeros(n1,n2,n3);
            bin_tensor4 = zeros(n1,9,7,n3);
            for i1 = 1:n1
                for i2 = 1:9
                    bin_tensor4(i1,i2,:,:) = round(rand_ten_slice(i1,i2)+0.5-missing_rate);
                end
            end
            for i1 = 1:n1
                for i2 = 1:n2
                    bin_tensor(i1,i2,:) = reshape(bin_tensor4(i1,ceil(i2/7),...
                        i2-7*(ceil(i2/7)-1),:),[n3,1]);
                end
            end
            sparse_tensor = original_tensor.*bin_tensor;
        elseif d == 4
            tensor = zeros(n1,9,7,n3);
            bin_tensor = zeros(n1,9,7,n3);
            for i1 = 1:n1
                for i2 = 1:n2
                    tensor(i1,ceil(i2/7),i2-7*(ceil(i2/7)-1),:) = ...
                        reshape(original_tensor(i1,i2,:),[n3,1]);
                end
            end
            for i1 = 1:n1
                for i2 = 1:9
                    bin_tensor(i1,i2,:,:) = round(rand_ten_slice(i1,i2)+0.5-missing_rate);
                end
            end
            sparse_tensor = tensor.*bin_tensor;
            original_tensor = tensor;
        end
end