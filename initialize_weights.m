function [vec_theta, hiddenSize] = initialize_weights(imageDim, layer, numClasses)

num_layer = size(layer,2);

% Calculate hiddenSize ----------------------------------------------------
Actual_dim = imageDim;
for k = 1:num_layer
    if strcmp(layer(k).name, 'Conv')
        Actual_numFilters = layer(k).numFilters;
        Actual_dim = (((Actual_dim + (2*layer(k).Padding) - layer(k).filterDim)/ layer(k).Stride) + 1 );
    elseif strcmp(layer(k).name, 'Activation')
        Actual_dim = Actual_dim;    
    elseif strcmp(layer(k).name, 'Pooling')
        Actual_dim = Actual_dim/(layer(k).Size);  
    else
        break
    end
end

hiddenSize = (Actual_dim^2)*Actual_numFilters;

% Create Theta ------------------------------------------------------------
Channels = layer(1).channels;
for k = 1:num_layer
    if strcmp(layer(k).name, 'Conv')
        theta(k).name = 'Conv';
        theta(k).weights = 1e-1*randn(layer(k).filterDim, layer(k).filterDim, Channels, layer(k).numFilters);
        theta(k).bias = zeros(layer(k).numFilters, 1);
        Channels = layer(k).numFilters;
    elseif strcmp(layer(k).name , 'FC')
        theta(k).name = 'FC';
        r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
        theta(k).weights = rand(numClasses, hiddenSize) * 2 * r - r;
        theta(k).bias = zeros(numClasses, 1);
    end
end

% Vectorize Theta ------------------------------------------------------------
%vec_theta = [theta(1).weights(:) ; theta(4).weights(:) ; theta(1).bias(:) ; theta(4).bias(:)];

vec_theta = [];
for k = 1:num_layer
    if strcmp(layer(k).name, 'Conv')
        vec_theta = [vec_theta(:); theta(k).weights(:)];
    elseif strcmp(layer(k).name , 'FC')
        vec_theta = [vec_theta(:); theta(k).weights(:)];
    end
end

for k = 1:num_layer
    if strcmp(layer(k).name, 'Conv')
        vec_theta = [vec_theta(:); theta(k).bias(:)];
    elseif strcmp(layer(k).name , 'FC')
        vec_theta = [vec_theta(:); theta(k).bias(:)];
    end
end

end

%---------------------------------------------------------------------------

% function theta = initialize_weights(imageDim,filterDim,numFilters,Stride,Padding,...
%                                 poolDim,numClasses)
% % Initialize parameters for a single layer convolutional neural
% % network followed by a softmax layer.
% %                            
% % Parameters:
% %  imageDim   -  height/width of image
% %  filterDim  -  dimension of convolutional filter                            
% %  numFilters -  number of convolutional filters
% %  poolDim    -  dimension of pooling area
% %  numClasses -  number of classes to predict
% %
% %
% % Returns:
% %  theta      -  unrolled parameter vector with initialized weights
% 
% %% Initialize parameters randomly based on layer sizes.
% assert(filterDim < imageDim,'filterDim must be less that imageDim');
% 
% Wc = 1e-1*randn(filterDim,filterDim,numFilters);
% 
% outDim = ((imageDim + (2*Padding) - filterDim)/Stride) + 1; % dimension of convolved image
% 
% % assume outDim is multiple of poolDim
% assert(mod(outDim,poolDim)==0,...
%        'poolDim must divide imageDim - filterDim + 1');
% 
% outDim = outDim/poolDim;
% hiddenSize = outDim^2*numFilters;
% 
% % we'll choose weights uniformly from the interval [-r, r]
% r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
% Wd = rand(numClasses, hiddenSize) * 2 * r - r;
% 
% bc = zeros(numFilters, 1);
% bd = zeros(numClasses, 1);
% 
% % Convert weights and bias gradients to the vector form.
% % This step will "unroll" (flatten and concatenate together) all 
% % your parameters into a vector, which can then be used with minFunc. 
% theta = [Wc(:) ; Wd(:) ; bc(:) ; bd(:)];
% 
% end