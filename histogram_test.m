%% Written by Muhammet Balcilar , France
% All rights reserved

clear all
close all

% define class names in the data folder
categories = {'kitchen', 'store', 'bedroom', 'livingroom', 'house', ...
       'industrial', 'stadium', 'underwater', 'tallbuilding', 'street', ...
       'highway', 'field', 'coast', 'mountain', 'forest'};  
   
% import vl library for dense sift calculations
run vlfeat-0.9.21/toolbox/vl_setup
 
% read train and test images class name and file names
trainingImages = imageDatastore('data/train','IncludeSubfolders',true,'LabelSource','foldernames');
testImages = imageDatastore('data/test','IncludeSubfolders',true,'LabelSource','foldernames');
  
% define paramters of histogram features
colorspace='hsv';
nbin=20;

% extract train set features
for i=1:length(trainingImages.Files)
    fprintf('Train image feature extraction:%f\n',i/length(trainingImages.Files));
    % read image file name
    fname=trainingImages.Files(i);
    % calculate its histogram features and read actual class name
    train_image_feats(i,:)=get_colour_histograms(fname,colorspace,nbin);
    train_labels(i,1)=trainingImages.Labels(i);
end

% extract test set features
for i=1:length(testImages.Files)
    fprintf('Test image feature extraction:%f\n',i/length(testImages.Files));
    % read image file name
    fname=testImages.Files(i);
    % calculate its histogram features and read actual class name
    test_image_feats(i,:)=get_colour_histograms(fname,colorspace,nbin);
    test_labels(i,1)=testImages.Labels(i);
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
msg=['Histogram''s accuracy= ' num2str(accuracy)];
title(msg);
colorbar


figure;plotroc(targets,scores')


