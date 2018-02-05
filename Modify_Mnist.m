function [ New_images] = Modify_Mnist( Images, makeodd )

[R, C, M] = size(Images);

if makeodd == 1
    New_images = zeros(R+1,C+1,1,M);
    for m = 1 : M
        pad_image = padarray(Images(:,:,m),[1 1]);
        New_images(:,:,1,m) = pad_image(1:end-1, 1:end-1);
    end
else
    New_images = zeros(R,C,1,M);
    for m = 1 : M
        New_images(:,:,1,m) = Images(:,:,m);
    end    
end

end

