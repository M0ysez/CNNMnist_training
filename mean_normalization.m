function [ norm_input ] = mean_normalization( input )

data_mean = mean(input, 1);
data_std = std(input, 1);
data_std(data_std == 0) = 1;
norm_input = bsxfun(@minus, input, data_mean);

norm_input = bsxfun(@rdivide, norm_input, data_std);


% data_mean = mean(input, 2);
% data_std = std(input, 0, 2);
% data_std(data_std == 0) = 1;
% norm_input = bsxfun(@minus, input, data_mean);
% 
% norm_input = bsxfun(@rdivide, norm_input, data_std)
end

% [n,m] = size(input);
% norm_input = zeros(n,m);
% 
% for k = 1:n
%     feature = input(k,:);
%     norm_input(k,:) = (feature - mean(feature))./ std(feature);
% end
