
# Deep Sceen Classification

Scene classification with using some certain different images of scenes are very important and crucial issue in computer vision literature. Especially, to automatize it with computer has significant amount of benefit in terms of robotic and automation. Although computers are still far from the human beings ability in order to visual understanding, the researchers have done too many significant contribution on this area. As general classification problem, image scene classification problem has the same two fundamental step in it. These are feature extraction and classification respectively. 

Feature extraction step consist of to figure out how the scene image has to be represented. If we do analogy with human visual understanding, human brain codes and indexes the image with some certain features. Let us assume how we recognize an apple as soon as we see it. Our brain has some model as we called it “being an apple”. Under this model, our brain represents this information with some unknown information. Maybe its color, shape, smell, or some other things which we have never though it before. With the same analogy, our computers needs to represent any model with some numeric values which we called it “features”. These features has to be invariant too many different real world problems, such as scale invariant, rotational invariant, illumination invariant. We really are be able to distinguish an apple even low illumination environment, or if we see it different angle or even too close or enough degree far. That means human being “being an apple” features are scale-rotation-illumination invariant. 

Classification step consist of how the presented features far or similar from compared model. This step can be used to analogy with human being too. For instance f we do not take enough evidence of any fruit and we have a doubt that if it is an apple or orange. Regardless of having doubt or not, out classification mechanism takes action and evaluates collected evidence then makes final decision on it. Such like this, computers has to have this kind of evidence compare mechanism which we called it classification.

In this research, we have focused on three different image features which are histogram of color, bag of sift features and finally convolution neural networks. For classification method, we have just tried multi class support vector machine. Our research shows that pretrained convolutional neural network named vgg16 has reasonable accuracy.


## Reference ##
[1]	Shi, Qinfeng, et al. "Is face recognition really a compressive sensing problem?." Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on. IEEE, 2011. 
