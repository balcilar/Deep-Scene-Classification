%% Written by Muhammet Balcilar , France
% All rights reserved

clear all
close all

% define class names in the data folder
categories = {'kitchen', 'store', 'bedroom', 'livingroom', 'house', ...
   'industrial', 'stadium', 'underwater', 'tallbuilding', 'street', ...
   'highway', 'field', 'coast', 'mountain', 'forest'}; 




% load pretrained network
net = vgg16;

% take input size of network which our images has to be in that resolution
inputSize = net.Layers(1).InputSize;

% allocate matrix for train and test features and labels
train_image_feats=[];
train_labels=[];

test_image_feats=[];
test_labels=[];

% to avoid memory problem, we read all category images step by step instead
% of reading all. since the vgg16 network is too big it is not possible to
% read all image and calculate features using ordinary PC.

for i=1:length(categories)
    % now we read and calculate features of image in just current category 
    % there are 100 train and 100 test images for every single category
    categories{i}
    % read train and test images in current category
    trainingImages = imageDatastore(['data/train/' categories{i}],'IncludeSubfolders',true,'LabelSource','foldernames');
    testImages = imageDatastore(['data/test/' categories{i}],'IncludeSubfolders',true,'LabelSource','foldernames');

    % get the images in same resolution with network inputs.
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingImages);
    augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);

    % vgg16 networks has 41 layer. we used 39th layer output as features.
    % to take the features we used activation function to learn fc8 layer
    % (39th later) output as features set.

    trainingFeatures = activations(net,augimdsTrain,'fc8','OutputAs','channels');
    testFeatures = activations(net,augimdsTest,'fc8','OutputAs','channels');

    % accumulate all features and its label into the global matrix.
    train_image_feats=[train_image_feats;squeeze(trainingFeatures(1,1,:,:))'];    
    train_labels=[train_labels;trainingImages.Labels];

    test_image_feats=[test_image_feats;squeeze(testFeatures(1,1,:,:))'];
    test_labels=[test_labels;testImages.Labels];
end




svm = fitcecoc(train_image_feats,train_labels);

[predictedLabels scores]=predict(svm,test_image_feats);


% calculate accuracy
accuracy = mean(predictedLabels == test_labels);

targets=zeros(size(scores'));

for i=1:size(test_labels,1)
I=find(categories==test_labels(i));
targets(I(1),i)=1;
end

cmat=confusionmat(test_labels,predictedLabels);

figure;imagesc(cmat);
xlabel('Target Class');
ylabel('Predicted Class');
msg=['VGG16 network''s accuracy= ' num2str(accuracy)];
title(msg);
colorbar


figure;plotroc(targets,scores')
















