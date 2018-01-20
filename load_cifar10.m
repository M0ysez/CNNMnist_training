function [Data,labels,meta] = load_cifar10(rawset)

set = load(rawset);
meta = load('batches.meta.mat');
meta = meta.label_names;

labels = set.labels;

[M, ~] =size(set.data); 

im=zeros(32,32,3);
Data=zeros(32,32,3,M);

for m=1:M 
    
    R = set.data(m,1:1024);
    G = set.data(m,1025:2048);
    B = set.data(m,2049:3072);

    k=1;
    for x=1:32
        for i=1:32
          im(x,i,1)=R(k);
          im(x,i,2)=G(k);
          im(x,i,3)=B(k);
          k=k+1;
        end
    end  

Data(:,:,:,m)=double(im)/255;

end

% figure()
% for k = 1:25
% subplot(5,5,k)
% imshow(Data(:,:,:,k))
% title(strcat(num2str(set.labels(k)),' ',meta(labels(k)+1)))
% end

end