# Skin-Disease-Identification-using-YOLO

# YOLO

You Only Look Once is a state-of-the-art, real-time object detection system. It was originally developed around 2015 and outperformed every other technique at that time.
YOLO has its own neat architecture based on CNN and anchor boxes and is proven to be an on-the-go object detection technique for widely used problems. With the timeline, it has become faster and better, with its versions named as:
●	YOLO V1
●	YOLO V2
●	YOLO V3
YOLO V2 is better than V1 in terms of accuracy and speed.
YOLO V3 is not faster than V2 but is more accurate than V2.

# Skin disease identification

This Recognition system is built using YOLO Object detection algorithm. By considering three categories of skin disease such as Psoriasis, Rosacea and Melanoma.

# Dataset :

For Building the Skin recognition model 200 images were collected for each class, so totally 600 images were used from various resources like ImageNet, Kaggle and Google Image.
The images were preprocessed and categorized into respective categories.
The total dataset was splitted into two parts i.e 80% (160 Images) of the Dataset for training the model and 20% (40 Images) for testing the model.

# Approach
Darknet is the official implementation of Yolo object algorithm
Darknet (https://pjreddie.com/darknet/). This is the “official” implementation, created by the same people behind the algorithm. It is written in C with CUDA, hence it supports GPU computation. It is actually a complete neural network framework, so it really can be used for other objectives besides YOLO detection. 


# How Does YOLO Work?

 
The first step to understanding YOLO is how it encodes its output. The input image is divided into an S x S grid of cells. For each object that is present on the image, one grid cell is said to be “responsible” for predicting it. That is the cell where the center of the object falls into.
Each grid cell predicts B bounding boxes as well as C class probabilities. 

The bounding box prediction has 5 components: (x, y, w, h, confidence). The (x, y) coordinates represent the center of the box, relative to the grid cell location (remember that, if the center of the box does not fall inside the grid cell, then this cell is not responsible for it). These coordinates are normalized to fall between 0 and 1. The (w, h) box dimensions are also normalized to [0, 1], relative to the image size.

Note that the confidence reflects the presence or absence of an object of any class. In case you don't know what IOU is, take a look here.
Now that we understand the 5 components of the box prediction, remember that each grid cell makes B of those predictions, so there are in total S x S x B * 5 outputs related to bounding box predictions.

It is also necessary to predict the class probabilities, Pr(Class(i) | Object). This probability is conditioned on the grid cell containing one object (see this if you don’t know that conditional probability means). In practice, it means that if no object is present on the grid cell, the loss function will not penalize it for a wrong class prediction, as we will see later. The network only predicts one set of class probabilities per cell, regardless of the number of boxes B. That makes S x S x C class probabilities in total
Adding the class predictions to the output vector, we get a S x S x (B * 5 +C) tensor as output.

# Intersection Over Union Threshold(IOU)

After removing all the predicted bounding boxes that have a low detection probability, the second step in NMS, is to select the bounding boxes with the highest detection probability and eliminate all the bounding boxes whose Intersection Over Union (IOU) value is higher than a given IOU threshold. In the code below, we set this IOU threshold to 0.4. This means that all predicted bounding boxes that have an IOU value greater than 0.4 with respect to the best bounding boxes will be removed.
 

# Non-Maximal Suppression (NMS)

YOLO uses Non-Maximal Suppression (NMS) to only keep the best bounding box. The first step in NMS is to remove all the predicted bounding boxes that have a detection probability that is less than a given NMS threshold. In the code below, we set this NMS threshold to 0.6. This means that all predicted bounding boxes that have a detection probability less than 0.6 will be removed.
 

 

 
