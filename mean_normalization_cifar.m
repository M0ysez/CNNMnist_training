function [ New_images ] = mean_normalization_cifar( Images )

[R, C, CH ,M] = size(Images);

New_images = zeros(R,C,CH,M);
Images3d = zeros(R,C,CH,M);
for ch = 1 : CH
    for m = 1 : M
        Images3d(:,:,m) = Images(:,:,ch,m);
    end
    
    Images3d_norm = mean_normalization(Images3d);

    for m = 1 : M
       New_images(:,:,ch,m) = Images3d_norm(:,:,m);
    end
end

end

