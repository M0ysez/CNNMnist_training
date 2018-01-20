function [cost, grad, preds] = cnn_cost(theta,images,labels,numClasses,...
    layer, hiddenSize, pred)

if ~exist('pred','var')
    pred = false;
end;

imageDim = size(images,1); % height/width of image
numImages = size(images,4); % number of images
num_layer = size(layer,2);

weightDecay = 1e-3; % regularization
USE_WEIGHT_DECAY = 1;

Activations  = Create_activations(imageDim, layer, numImages, numClasses );
[weights] = cnnParamsToStack(theta, layer, hiddenSize, numClasses);

%%======================================================================
% Forward Propagation

 Rotated_weights  = Rotate_weights( layer, weights );

for imageNum = 1 : numImages
    image = images(:, :, :,imageNum);
    
    for k = 1:num_layer-1
            if strcmp(layer(k).name, 'Conv')
                numFilters = layer(k).numFilters;
                for filterNum = 1 : numFilters
                    if k == 1 
                        channels = size(image,3);
                        for ch = 1 : channels
                            filteredImage = conv2(image(:,:,ch), Rotated_weights{k,1}(:, :, ch,filterNum), 'valid') ;
                            filteredImage = Subsample_Stride( filteredImage, layer(k).Stride );
                            Activations{k}(:, :, filterNum, imageNum) = Activations{k}(:, :, filterNum, imageNum) + filteredImage;
                        end
                        Activations{k}(:, :, filterNum, imageNum) =  Activations{k}(:, :, filterNum, imageNum) + weights{k,2}(filterNum);
                    else
                        channels = size(Activations{k-1},3);
                        for ch = 1 : channels
                            filteredImage =  conv2( Activations{k-1}(:,:,ch,imageNum), Rotated_weights{k,1}(:, :, ch,filterNum), 'valid') ;
                            filteredImage =  Subsample_Stride( filteredImage, layer(k).Stride ); 
                            Activations{k}(:, :, filterNum, imageNum) =  Activations{k}(:, :, filterNum, imageNum) + filteredImage;
                        end
                        Activations{k}(:, :, filterNum, imageNum) = Activations{k}(:, :, filterNum, imageNum)+ weights{k,2}(filterNum);
                    end    
                    
                end
                conv_layer_num = k ; 
            elseif strcmp(layer(k).name , 'Activation')
                activationType = layer(k).type;
                for filterNum = 1 : numFilters
                    switch activationType
                        case 'relu'
                            Activations{k}(:, :, filterNum, imageNum) = max(Activations{k-1}(:, :, filterNum, imageNum), 0); % relu
                        case 'sigmoid'
                            Activations{k}(:, :, filterNum, imageNum) = sigmoid(Activations{k-1}(:, :, filterNum, imageNum)); % sigmoid
                    end
                end    
            elseif strcmp(layer(k).name , 'Pooling')
                poolDim = layer(k).Size;
                meanPoolingFilter = ones(poolDim, poolDim);
                areaOfPoolingFilter = poolDim ^ 2;
                meanPoolingFilter = meanPoolingFilter / areaOfPoolingFilter;
                
                poolingIndex = 1 : poolDim :  (size(Activations{k-1},1));
                for filterNum = 1 : numFilters
                    pooledImage = conv2(Activations{k-1}(:, :, filterNum, imageNum), meanPoolingFilter, 'valid');
                    Activations{k}(:, :, filterNum, imageNum) = pooledImage(poolingIndex, poolingIndex);
                end
                  %activationsPooled= Activations{k};
                 inputlayer_num = k;
            end
    end     
end

activationsPooledReshaped = reshape(Activations{inputlayer_num},[],numImages);
probs = zeros(numClasses,numImages);

activationsSoftmax = weights{num_layer,1} * activationsPooledReshaped + repmat(weights{num_layer,2}, 1, numImages);
activationsSoftmax = bsxfun(@minus, activationsSoftmax, max(activationsSoftmax));
activationsSoftmax = exp(activationsSoftmax);
probs = bsxfun(@rdivide, activationsSoftmax, sum(activationsSoftmax));

%%======================================================================
% Calculate Cost

cost = 0; 

labelIndex = sub2ind(size(activationsSoftmax), labels', 1:numImages);
onehotLabels = zeros(size(activationsSoftmax));
onehotLabels(labelIndex) = 1;
cost = -sum(sum(onehotLabels .* log(probs)));

weightDecayCost_conv = 0;
if USE_WEIGHT_DECAY
    for k = 1:num_layer
        if strcmp(layer(k).name, 'Conv')
            weightDecayCost_conv = weightDecayCost_conv + sum(weights{k,1}(:) .^ 2);
        elseif strcmp(layer(k).name, 'Activation')

        elseif strcmp(layer(k).name, 'Pooling')

        else
            weightDecayCost_conv = weightDecayCost_conv + sum(weights{k,1}(:) .^ 2);
        end
    end
    weightDecayCost = .5 * weightDecay *  weightDecayCost_conv;
else
    weightDecayCost = 0;
end
cost = cost / numImages + weightDecayCost;

if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end

%%======================================================================
% Backpropagation

errorsSoftmax = probs - onehotLabels;
errorsSoftmax = errorsSoftmax / numImages;

Grads{inputlayer_num+1,1} = errorsSoftmax * activationsPooledReshaped';
if USE_WEIGHT_DECAY
    Grads{inputlayer_num+1,1} = Grads{inputlayer_num+1,1} + weightDecay * weights{inputlayer_num+1,1};
end
 Grads{inputlayer_num+1,2} = sum(errorsSoftmax, 2);

inputnn_dim = size(Activations{inputlayer_num},1);

errorsPooled = weights{num_layer,1}' * errorsSoftmax;
errorsPooled = reshape(errorsPooled, [], inputnn_dim, numFilters, numImages);

Error{inputlayer_num + 1} = errorsPooled;
for k = fliplr(1:inputlayer_num)
    
    if strcmp(layer(k).name, 'Pooling')
        numFilters = size(Error{k+1},3);
        errorsPooling = zeros( size(Error{k+1},1)*layer(k).Size , size(Error{k+1},1)*layer(k).Size , numFilters, numImages);
        unpoolingFilter = ones(layer(k).Size);

        poolArea = layer(k).Size ^ 2;
        unpoolingFilter = unpoolingFilter / poolArea;
        parfor imageNum = 1:numImages
            % for imageNum = 1:numImages
            for filterNum = 1:numFilters
                e = Error{k+1}(:, :, filterNum, imageNum);
                errorsPooling(:, :, filterNum, imageNum) = kron(e, unpoolingFilter);
            end
        end
      Error{k} = errorsPooling; 
      
     elseif strcmp(layer(k).name , 'Activation') 
     activationType = layer(k).type; 
     switch activationType
            case 'relu'
                errorsConvolution = Error{k+1} .* (Activations{k} > 0); % relu derivative = x > 1
            case 'sigmoid'
                errorsConvolution = Error{k+1} .* Activations{k} .* (1 - Activations{k}); % sigmoid derivative = x .* (1 - x)
     end
     Error{k} = errorsConvolution;
     
     elseif strcmp(layer(k).name , 'Conv')
     numFilters = layer(k).numFilters;

       Grads{k,2} = zeros(size(weights{k,2}));
       Grads{k,1} = zeros(size(weights{k,1}));     

        % parfor filterNum = 1 : numFilters
        for filterNum = 1 : numFilters
            e =  Error{k+2}(:, :, filterNum, :);
            Grads{k,2}(filterNum) = sum(e(:));
        end
        for filterNum = 1 : numFilters
            % for filterNum = 1 : numFilters
            for imageNum = 1 : numImages
                e = Error{k+1}(:, :, filterNum, imageNum);
                %         e = errorsPooling(:, :, filterNum, imageNum);
                errorsConvolution(:, :, filterNum, imageNum) = rot90(e, 2);
            end
        end
       
        ErrorConv = 0;
       for filterNum = 1 : numFilters
           for ch = 1 : size(weights{k},3)
           ErrorConv = ErrorConv  + conv2(weights{k}(:, :,ch, filterNum), errorsConvolution(:, :, filterNum, imageNum), 'full');   
           end
           Error{k}(:,:,ch,imageNum) = ErrorConv;
       end 
        
       for filterNum = 1 : numFilters
            Wc_gradFilter = zeros(size(Grads{k,1}, 1), size(Grads{k,1}, 2));
            %     parfor imageNum = 1 : numImages
            for imageNum = 1 : numImages
                if k == 1
                    for ch = 1 : layer(k).channels 
                        Wc_gradFilter = Wc_gradFilter + conv2(images(:, :, ch,imageNum), errorsConvolution(:, :, filterNum, imageNum), 'valid');
                        Grads{k,1}(:, :, ch ,filterNum) = Wc_gradFilter;
                    end
                else
                    for ch = 1 : size(Activations{k-1},3)
                       Wc_gradFilter = Wc_gradFilter + conv2(Activations{k-1}(:, :,ch, imageNum), errorsConvolution(:, :, filterNum, imageNum), 'valid');   
                       Grads{k,1}(:, :, ch ,filterNum) = Wc_gradFilter;
                    end
                    
                end    
            end
        
        end
        if USE_WEIGHT_DECAY
            Grads{k,1} = Grads{k,1} + weightDecay * weights{k,1};
        end 
    
    end
    
end

grad = [];
for k = 1:num_layer
    if strcmp(layer(k).name, 'Conv')
        grad = [grad(:); Grads{k,1}(:)];
    elseif strcmp(layer(k).name , 'FC')
        grad = [grad(:); Grads{k,1}(:)];
    end
end

for k = 1:num_layer
    if strcmp(layer(k).name, 'Conv')
        grad = [grad(:); Grads{k,2}(:)];
    elseif strcmp(layer(k).name , 'FC')
        grad = [grad(:); Grads{k,2}(:)];
    end
end

end
