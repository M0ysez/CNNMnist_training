function [ New_images] = Modify_Mnist( Images )

[R, C, M] = size(Images);
New_images = zeros(R,C,1,M);
for m = 1 : M
    New_images(:,:,1,m) = Images(:,:,m);
end

end

