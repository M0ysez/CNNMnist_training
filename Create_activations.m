function [ Activations ] = Create_activations(imageDim, layer, numImages, numClasses )

num_layer = size(layer,2);

Activation_dim = imageDim;
for k = 1:num_layer
    if strcmp(layer(k).name, 'Conv')
        Activation_dim =  ((Activation_dim + (2*layer(k).Padding) - layer(k).filterDim)/layer(k).Stride) + 1;
        Activations{k} = zeros(Activation_dim, Activation_dim, layer(k).numFilters, numImages);
        numfilters = layer(k).numFilters;
    elseif strcmp(layer(k).name , 'Activation')
        Activations{k} = zeros(Activation_dim, Activation_dim, numfilters, numImages);
    elseif strcmp(layer(k).name , 'Pooling')    
        Activation_dim =  (Activation_dim)/layer(k).Size;
        Activations{k} = zeros(Activation_dim, Activation_dim, numfilters, numImages);
    elseif strcmp(layer(k).name , 'FC')
        Activations{k} = zeros(1,numClasses);
    end
end

end

