% function [Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,...
%                                  numFilters,poolDim,numClasses)

function [weights] = cnnParamsToStack(theta, layer, hiddenSize, numClasses)

num_layer = size(layer,2);
ind_inicial = 0;
for k = 1:num_layer
    if strcmp(layer(k).name, 'Conv')
        if k == 1
        ind_final = layer(k).filterDim * layer(k).filterDim * layer(k).channels * layer(k).numFilters + ind_inicial;
        weights{k,1} = {theta(ind_inicial+1:ind_final)};
        weights{k,1} = reshape(weights{k,1}{1},layer(k).filterDim , layer(k).filterDim, layer(k).channels, layer(k).numFilters );
        else
        ind_final = layer(k).filterDim * layer(k).filterDim * numchannels * layer(k).numFilters + ind_inicial; 
        weights{k,1} = {theta(ind_inicial+1:ind_final)};
        weights{k,1} = reshape(weights{k,1}{1},layer(k).filterDim , layer(k).filterDim, numchannels, layer(k).numFilters );
        end           
        ind_inicial = ind_final;
        numchannels = layer(k).numFilters;
    elseif strcmp(layer(k).name , 'FC')
        ind_final = hiddenSize * numClasses + ind_inicial;
        weights{k,1} = {theta(ind_inicial+1:ind_final)};
        weights{k,1} = reshape( weights{k,1}{1}, numClasses, hiddenSize);
        ind_inicial = ind_final;
    end
end

for k = 1:num_layer
    if strcmp(layer(k).name, 'Conv')
        ind_final = layer(k).numFilters + ind_inicial;
        weights{k,2} = {theta(ind_inicial+1:ind_final)};
        weights{k,2} = weights{k,2}{1} ;
        ind_inicial = ind_final;
    elseif strcmp(layer(k).name , 'FC')
        ind_final = numClasses + ind_inicial;
        weights{k,2} = {theta(ind_inicial+1:ind_final)};
        weights{k,2} = weights{k,2}{1};
        ind_inicial = ind_final;
    end
end


% % Converts unrolled parameters for a single layer convolutional neural
% % network followed by a softmax layer into structured weight
% % tensors/matrices and corresponding biases
% %                            
% % Parameters:
% %  theta      -  unrolled parameter vectore
% %  imageDim   -  height/width of image
% %  filterDim  -  dimension of convolutional filter                            
% %  numFilters -  number of convolutional filters
% %  poolDim    -  dimension of pooling area
% %  numClasses -  number of classes to predict
% %
% %
% % Returns:
% %  Wc      -  filterDim x filterDim x numFilters parameter matrix
% %  Wd      -  numClasses x hiddenSize parameter matrix, hiddenSize is
% %             calculated as numFilters*((imageDim-filterDim+1)/poolDim)^2 
% %  bc      -  bias for convolution layer of size numFilters x 1
% %  bd      -  bias for dense layer of size hiddenSize x 1
% 
% outDim = (imageDim - filterDim + 1)/poolDim;
% hiddenSize = outDim^2*numFilters;
% 
% %% Reshape theta
% indS = 1;
% indE = filterDim^2*numFilters;
% Wc = reshape(theta(indS:indE),filterDim,filterDim,numFilters);
% indS = indE+1;
% indE = indE+hiddenSize*numClasses;
% Wd = reshape(theta(indS:indE),numClasses,hiddenSize);
% indS = indE+1;
% indE = indE+numFilters;
% bc = theta(indS:indE);
% bd = theta(indE+1:end);


end