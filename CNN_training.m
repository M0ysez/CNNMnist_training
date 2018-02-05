%% Convolution Neural Network Training
clear all
clc
addpath D:\Datasets\Mnist
addpath D:\Datasets\Caltech_silhouettes
addpath D:\Datasets\Cifar\cifar-10-batches-mat
%% ======================================================================
%Cargar conjunto de entrenamiento y prueba

load('caltech101_silhouettes_28.mat')
images = reshape(X',28,28,[]);
images = mean_normalization(images);
images = Modify_Mnist( images,0);
labels = Y';
testImages = images;
testLabels = labels;
%-------------------------------------------------------------------
% [images,labels,~] = load_cifar10('data_batch_1.mat');
% images = mean_normalization_cifar( images); 
% labels(labels==0) = 10; % Remap 0 to 10
% 
% [testImages,testLabels,meta] = load_cifar10('test_batch.mat');
% testImages = mean_normalization_cifar( testImages); 
% testLabels(testLabels==0) = 10; % Remap 0 to 10

%-------------------------------------------------------------------
% images = loadMNISTImages('train-images.idx3-ubyte');
% images = reshape(images,28,28,[]);
% images = mean_normalization(images);
% images = Modify_Mnist( images,0);
% labels = loadMNISTLabels('train-labels.idx1-ubyte');
% labels(labels==0) = 10; % Remap 0 to 10
% 
% testImages = loadMNISTImages('t10k-images.idx3-ubyte');
% testImages = reshape(testImages,28,28,[]);
% testImages = mean_normalization(testImages);
% testImages = Modify_Mnist( testImages,0 );
% testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
% testLabels(testLabels==0) = 10; % Remap 0 to 10
%-------------------------------------------------------------------

imageDim = size(images,1);
numClasses = max(labels);  % Number of classes (MNIST images fall into 10 classes)

%% ======================================================================
% Crear Red neuronal de convolucion
capa = 1;
layer(capa).name = 'Conv';
layer(capa).channels = size(images,3);
layer(capa).filterDim = 9;
layer(capa).numFilters = 50;
layer(capa).Stride = 1;
layer(capa).Padding = 0;

capa = capa + 1;
layer(capa).name = 'Activation';
layer(capa).type = 'relu';

% capa = capa + 1;
% layer(capa).name = 'Conv';
% layer(capa).filterDim = 3;
% layer(capa).numFilters = 20;
% layer(capa).Stride = 1;
% layer(capa).Padding = 0;
% 
% capa = capa + 1;
% layer(capa).name = 'Activation';
% layer(capa).type = 'relu';

% capa = capa + 1;
% layer(capa).name = 'Conv';

% layer(capa).filterDim = 3;
% layer(capa).numFilters = 25;
% layer(capa).Stride = 1;
% layer(capa).Padding = 0;
% 
% capa = capa + 1;
% layer(capa).name = 'Activation';
% layer(capa).type = 'relu';

capa = capa + 1;
layer(capa).name = 'Pooling';
layer(capa).type = 'Mean';
layer(capa).Size = 2;
layer(capa).Stride = 2;

% capa = capa + 1;
% layer(capa).name = 'Conv';
% layer(capa).filterDim = 5;
% layer(capa).numFilters = 20;
% layer(capa).Stride = 1;
% layer(capa).Padding = 0;

% capa = capa + 1;
% layer(capa).name = 'Activation';
% layer(capa).type = 'relu';

% capa = capa + 1;
% layer(capa).name = 'Pooling';
% layer(capa).type = 'Mean';
% layer(capa).Size = 2;
% layer(capa).Stride = 2;

capa = capa + 1;
layer(capa).name = 'FC';
layer(capa).Hidden = [0];
layer(capa).OutputSize = numClasses;

% Initialize Parameters
[theta, hiddenSize] = initialize_weights(imageDim, layer, numClasses);

%%=================================================================
%Training
options.epochs = 3;
options.minibatch = 512;
options.alpha = 0.1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnn_cost(x,y,z,numClasses,layer, hiddenSize),...
                      theta,images,labels,options);

%%======================================================================
% Test

[~,cost,preds]=cnn_cost(opttheta,testImages,testLabels,numClasses,...
                layer, hiddenSize, true);

acc = sum(preds==testLabels)/length(preds);
fprintf('Accuracy is %f\n',acc);
