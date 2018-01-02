# YOLO-OBJECT-DETECTION
Real time object detection using the state of the art Deep Learning algorithm, YOLO implemented in Keras

Download the weights and save in the root folder from here:
https://drive.google.com/file/d/0B1tW_VtY7onibmdQWE1zVERxcjQ/view?usp=sharing


# How YOLO works?</br>
YOLO - You Only Look Once is state of the art real time object detecting algorithm. When run on GPU's we get astonishing frame-rates upto 200 FPS.
YOLO takes a completely different approach than traditional R-CNN's and Fast R-CNN's which first defines many possible region of interest and then predicting the bounding boxes in those regions. 
YOLO on the other hand, scans the entire image only once (hence the name) using the concept of convolution and hence is so efficient.</br>

# Input
The input image is first divided into S X S grid.</br>
![Input image](http://machinethink.net/images/yolo/Grid@2x.png)

Each grid is responsible for predicting B = 2 bounding boxes.</br>
Depending on the confidence for each box, if it exceeds the threshold value defined by us, we draw that bounding box.</br>
Here is the YOLO tiny network which was used here:</br>

![YOLO tiny](https://github.com/shivamsaboo17/CarND-Vehicle-Detection/blob/master/output_images/mode_yolo_plot.jpg)

# Output
The output is a single dimensional flattened vector of shape 1 x 1470:
![Output vector](https://github.com/shivamsaboo17/CarND-Vehicle-Detection/raw/master/output_images/net_output.png)

Using the output box coordinates and their confidence we draw the bounding box accordingly.


## Dependencies:
1) Tensorflow</br>
2) Keras
3) Matplotlib
4) Tensorflow
5) Theano
6) Open CV
7) Numpy
