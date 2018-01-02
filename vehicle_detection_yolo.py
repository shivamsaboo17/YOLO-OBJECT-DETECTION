import imageio
imageio.plugins.ffmpeg.download()

import numpy as np
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Flatten, Activation, Reshape
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import cv2
import glob
import matplotlib.pyplot as plt

keras.backend.set_image_dim_ordering('th')

# Defining the keras model for tiny yolo
# The model takes input as 448 x 448 x 3 color image and outputs a tensor:
# of length 1470 which is equivalent as:
# 1) Dividing the image into 7 x 7 grid
# 2) Each grid predicting 2 box coordinates
# 3) Each prediction is softmax of 10 classes
# 4) 2 confidence values for presence of object
# Hence 7 x 7 x ((2 x 4 box-coordinates) + (2 x 10 softmax prediction) + 2) = 1470
model = Sequential()
# Parameters of the Convolution2D layer is as follows:
# no of filters, dimension of the filter, dimension of the input, border mode, subsample
model.add(Convolution2D(16, 3, 3, input_shape = (3, 448, 448), border_mode = 'same', subsample = (1, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, border_mode = 'same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Convolution2D(64,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(128,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(256,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(512,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(4096))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1470))

# Import the utils helper code for plotting of boxes
from utils import load_weights, Box, yolo_net_out_to_car_boxes, draw_boxes

# Load weights from pretrained yolo weights



load_weights(model, 'yolo-tiny.weights')

# Apply model to test images
image_path = './test_images/test8.jpg'
image = plt.imread(image_path)
image_crop = image[300:650, 500:,:]
resized = cv2.resize(image_crop, (448, 448))

batch = np.transpose(resized,(2,0,1))
batch = 2*(batch/255.) - 1
batch = np.expand_dims(batch, axis=0)
out = model.predict(batch)

boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17 )
# Uncomment the plotting code in utils draw_boxes functions to see the bounding boxes
draw_boxes(boxes,plt.imread(image_path),[[500,1280],[300,650]])
