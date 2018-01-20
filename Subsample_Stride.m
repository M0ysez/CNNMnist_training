function [ Subsampled_images ] = Subsample_Stride( Images, Stride )

if Stride == 1
Subsampled_images = Images;
return
end

[F, ~, M] = size(Images);
convDim =  ceil(F/Stride);
Subsampled_images = zeros(convDim,convDim,M);

for map = 1:M
    cont_row = 1;
    for row = 1:convDim
        cont_column = 1;
        for column = 1:convDim
           Subsampled_images(row,column,map)=Images(cont_row, cont_column,map); 
           cont_column = cont_column + Stride;
        end
        cont_row = cont_row + Stride;
    end
end

end

