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

% bag of sift feature size
vocab_size = 200;


feats=[];
for i=1:length(trainingImages.Files)
    fprintf('Vocabulary extraction:%f\n',i/length(trainingImages.Files));
    % read image 
    fname=trainingImages.Files{i};
    I=imread(fname);
    % convert it into grayscale
    I=single(rgb2gray(I));
    % calculate sift descriptors
    [f, d] = vl_dsift(I,'step',16);  
    % accumalate sifts descrptor into feat matrix
    feats=[feats d];
end
% cluster features into  number of vocal_size cluster
[vocab, assignments] = vl_kmeans(double(feats), vocab_size);
vocab=vocab';



train_image_feats=zeros(length(trainingImages.Files),size(vocab,1));
for i=1:length(trainingImages.Files)
    fprintf('Train image feature extraction:%f\n',i/length(trainingImages.Files));
    % read image 
    fname=trainingImages.Files{i};
    I=imread(fname);
    I=single(rgb2gray(I));
    % calculate sift descriptors
    [f, d] = vl_dsift(I,'step',4);
    
    % calculate cluster of all descriptors in the image into the vocabulary matrix
    D = vl_alldist2(vocab',double(d));
    [a b]=min(D);
    % find histogram of bags
    h=hist(b,1:size(vocab,1));
    % set normalzied histogram of vocabulary as features
    train_image_feats(i,:)=h/sum(h);  
    % set train class
    train_labels(i,1)=trainingImages.Labels(i);
end

test_image_feats=zeros(length(testImages.Files),size(vocab,1));
for i=1:length(testImages.Files)
    fprintf('Test image feature extraction:%f\n',i/length(testImages.Files));
    % read image 
    fname=testImages.Files{i};
    I=imread(fname);
    I=single(rgb2gray(I));
    % calculate sift descriptors
    [f, d] = vl_dsift(I,'step',4);
    
    % calculate cluster of all descriptors in the image into the vocabulary matrix
    D = vl_alldist2(vocab',double(d));
    [a b]=min(D);
    % find histogram of bags
    h=hist(b,1:size(vocab,1));
    % set normalzied histogram of vocabulary as features
    test_image_feats(i,:)=h/sum(h);  
    % set test class
    test_labels(i,1)=testImages.Labels(i);
end

%svm = fitcecoc(train_image_feats,train_labels);

%[predictedLabels scores]=predict(svm,test_image_feats);
[predictedLabels scores] = svm_classify(train_image_feats, train_labels, test_image_feats,0.000001);



% calculate accuracy
accuracy = mean(predictedLabels == test_labels);

targets=zeros(size(scores));

for i=1:size(test_labels,1)
    I=find(categories==test_labels(i));
    targets(I(1),i)=1;
end

cmat=confusionmat(test_labels,predictedLabels);

figure;imagesc(cmat);
xlabel('Target Class');
ylabel('Predicted Class');
msg=['Bag of Sift''s accuracy= ' num2str(accuracy)];
title(msg);
colorbar


figure;plotroc(targets,scores)





        

    
        
  


            
