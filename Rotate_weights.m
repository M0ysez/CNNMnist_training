function [ Rotated_weights ] = Rotate_weights( layer, weights )

num_layer = size(layer,2);
Rotated_weights = weights;

for k = 1:num_layer
    if strcmp(layer(k).name, 'Conv')
        for filterNum = 1 : layer(k).numFilters
            for ch = 1 : size(weights{k,1},3)
                Rotated_weights{k,1}(:, :, ch,filterNum) = rot90(weights{k,1}(:, :, ch ,filterNum), 2);
            end
        end
    elseif strcmp(layer(k).name , 'FC')
        return
    end
end

end



