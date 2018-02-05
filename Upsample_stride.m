function [ Upsampled_image ] = Upsample_stride( Image, Stride, filterdim )

Dim_out = size(Image,1);

if Stride == 1
    Upsampled_image = Image;
    return
end
Dim_in = (Dim_out-1) * Stride + filterdim;
Upsampled_image = zeros(Dim_in - filterdim + 1);

ind = 1:Stride:Dim_out*Stride;

Upsampled_image(ind,ind)=Image;

end

