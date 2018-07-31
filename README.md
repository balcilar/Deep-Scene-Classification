
# Deep Sceen Classification

Scene classification with using some certain different images of scenes are very important and crucial issue in computer vision literature. Especially, to automatize it with computer has significant amount of benefit in terms of robotic and automation. Although computers are still far from the human beings ability in order to visual understanding, the researchers have done too many significant contribution on this area. As general classification problem, image scene classification problem has the same two fundamental step in it. These are feature extraction and classification respectively. 

Feature extraction step consist of to figure out how the scene image has to be represented. If we do analogy with human visual understanding, human brain codes and indexes the image with some certain features. Let us assume how we recognize an apple as soon as we see it. Our brain has some model as we called it “being an apple”. Under this model, our brain represents this information with some unknown information. Maybe its color, shape, smell, or some other things which we have never though it before. With the same analogy, our computers needs to represent any model with some numeric values which we called it “features”. These features has to be invariant too many different real world problems, such as scale invariant, rotational invariant, illumination invariant. We really are be able to distinguish an apple even low illumination environment, or if we see it different angle or even too close or enough degree far. That means human being “being an apple” features are scale-rotation-illumination invariant. 

Classification step consist of how the presented features far or similar from compared model. This step can be used to analogy with human being too. For instance f we do not take enough evidence of any fruit and we have a doubt that if it is an apple or orange. Regardless of having doubt or not, out classification mechanism takes action and evaluates collected evidence then makes final decision on it. Such like this, computers has to have this kind of evidence compare mechanism which we called it classification.

In this research, we have focused on three different image features which are histogram of color, bag of sift features and finally convolution neural networks. For classification method, we have just tried multi class support vector machine. Our research shows that pretrained convolutional neural network named vgg16 has reasonable accuracy.

## Dataset

Our dataset  is composed of fifteen scene categories: thirteen were provided by Fei-Fei and Perona [1] (eight of these were originally collected by Oliva and Torralba [2]), and two (industrial and store) were collected by [3]. As a result in that dataset, there are 15 different class and every single class has 100 images for train and 100 images for test. All images are RBG color images but has different resolutions. Here is one example of every single group.

![Sample image](Outputs/sampleinputs.bmp?raw=true "Title")

## Histogram Based Features

The first and very primitive used feature is histogram of image. To do that we briefly count every pixel seeing. To do that first we used number of bin. Since normal RGB or grayscale pixel values are in range of 0 to 255. But other color space might be different. So we have to determine standard number of bin regardless of range of color channel value. For instance if we gives number of bin as 75, it means all inputs has to be quantized in 75 different discrete values. After quantization we count each quantized values frequency [4]. In our implementation we gives opportunities to select color space as grayscale, rbg, hvi, lab and xyz color spaces. In addition to color space, we created options to select number of bins. 

## Bag of word of SIFT Implementation
 
SIFT features are wide used and well-known feature descriptor in image processing literature. In this research we used it with not just some special points. Instead of it, we used it as dense manner. It means we extract nearly all points in the image. Nearly all means we could not used it for all pixels because of memory and computational cost problem. Instead of that we just extract SIFT every 16x16 pixel region center. It means we divided image 16x16 sub block and for every block center we calculated SIFT descriptor for construction of dictionary. After that we implemented kmeans clustering and divided all descriptor in train set into predefined cluster number as e used 30,50,80,120 and 200. At the last step, we calculated every 4x4 sub block center’s SIFT descriptor and calculate which class is the most closed to that descriptor. So to extract features, we assigned 1 to number of class label to every 4x4 block of test and train image to define the features. As a result we should say calculated feature vector is the histogram of SIFT descriptor [3]. Every SIFT descriptor vector is 128 length. If we use number of 200 cluster, our feature vector for reach image becomes 200x128 matrix.

## Deep Learning : VGG16 Pretrained Neural Network

Nowadays, many researcher reported that convolutional neural networks outperformed nearly all of the challenges in pattern classification area. That is why we selected to use deep learning to test onto out dataset. The most problematic issue of the deep learning is number of training element and computational source we need. To get rid of this problem, genrally researcher select to pre-trained networks which was trained by millions of images and huge computational source and ready to use for our applications. But not they are ready to implement in all cases. In this research we selected to use VGG16 [5] which was trained milions of image before. This network consist of 41 layers. Last 3 layers are for classification the given image into 1000 different category. For that reason it is not directly useful for us. To get rid of this problem, we took the 39th layer inputs as a feature vector and we used SVM  to classfy this features. VGG16’s 39th layer output has 1000 output. For that reason our feature has 1000 dimensions.

## Classification

In order to determine in which group is the every single test element belongs, we used svm classification with liner kernel but different regularization parameter. Normally SVM was presented by two class classification problem. But we implemented it as one to all comparison. Since we have 15 different class, we used 15 different SVM, one for copare class1 to rest, other compare class2 againts rest, so on so forth.

## Results

The results for histogram based features, we tried both gry scale , RGB and HSV color space with different number of bins from 10 to 200. As a result we took the best accuract with HVS color space under 20 number of bins. Although the accuracy is about 30% , we could not say it could not work. Since there are 15 different class and as baseline method random selection methods has just 100/15=6.66% accuracy. So we would say, according to histogram features results, there are significant differences between random selection and this method. To run histogram based method, following script should be run.

```
> histogram_test
```
The following figure is histogram features confusion matrix and ROC curves.

![Sample image](Outputs/histcmat.bmp?raw=true "Title")

The second results were taken from bag of sift features under 200 number of vocabulary size. Since this methods computational cost is quite big, but also the accuracy is above the 65%. To run sift based method, following script should be run.

```
> sift_test
```
The following figure is the confusion matrix and ROC curve for bag of sift method.

![Sample image](Outputs/siftcmat.bmp?raw=true "Title")


Last but now least we presented the best methods accuracy with was taken by vgg16’s pretrained networks. The accuracy is more than 90% and it is pretty suitable for many automatization tasks. To run vgg16 network based method, following script should be run.

```
> vgg16_test
```
The following figure is the confusion matrix and ROC curve for vgg16 based method

![Sample image](Outputs/vggcmat.bmp?raw=true "Title")


## Reference ##
[1]	Shi, Qinfeng, et al. "Is face recognition really a compressive sensing problem?." Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on. IEEE, 2011. 
