
function [predicted_categories scores] = svm_classify(train_image_feats, train_labels, test_image_feats,lambda)
%% Written by Muhammet Balcilar , France
% All rights reserved

categories = unique(train_labels); 
num_categories = length(categories);

for i=1:num_categories
    L=-1*ones(length(train_labels),1);
    I = (categories(i)==train_labels);
    L(I)=1;    
    [W ,B, INFO, SCORES] = vl_svmtrain(train_image_feats', L',lambda);
    scores(i,:) = W'*test_image_feats' + B ;  
end

[a b]=max(scores);

for i=1:length(b)
    predicted_categories(i,1)=categories(b(i));
end

%scores=100*(scores-min(scores))./(max(scores)-min(scores));
