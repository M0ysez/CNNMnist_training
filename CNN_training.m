%% Convolution Neural Network Training
clear all
clc
addpath D:\Datasets\Mnist
addpath D:\Datasets\Cifar\cifar-10-batches-mat
%% =======================================================================
%Cargar conjunto de entrenamiento y prueba

% [images,labels,~] = load_cifar10('data_batch_1.mat');
% labels(labels==0) = 10; % Remap 0 to 10
% 
% [testImages,testLabels,meta] = load_cifar10('test_batch.mat');
% testLabels(testLabels==0) = 10; % Remap 0 to 10

imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)

images = loadMNISTImages('train-images.idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
images = Modify_Mnist( images );
labels = loadMNISTLabels('train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

testImages = loadMNISTImages('t10k-images.idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testImages = Modify_Mnist( testImages );
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

%% ======================================================================
% Crear Red neuronal de convolucion

layer(1).name = 'Conv';
layer(1).channels = size(images,3);
layer(1).filterDim = 5;
layer(1).numFilters = 6;
layer(1).Stride = 1;
layer(1).Padding = 0;

layer(2).name = 'Activation';
layer(2).type = 'relu';

layer(3).name = 'Pooling';
layer(3).type = 'Mean';
layer(3).Size = 2;
layer(3).Stride = 2;
% % 
layer(4).name = 'Conv';
layer(4).filterDim = 5;
layer(4).numFilters = 30;
layer(4).Stride = 1;
layer(4).Padding = 0;

layer(5).name = 'Activation';
layer(5).type = 'relu';

layer(6).name = 'Pooling';
layer(6).type = 'Mean';
layer(6).Size = 2;
layer(6).Stride = 2;

layer(7).name = 'FC';
layer(7).Hidden = [0];
layer(7).OutputSize = numClasses;

% Initialize Parameters
[theta, hiddenSize] = initialize_weights(imageDim, layer, numClasses);

%%=================================================================
%Training
options.epochs = 5;
options.minibatch = 256;
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
