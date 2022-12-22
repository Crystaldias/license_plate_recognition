######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
#import matplotlib.pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'car1.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.80)


#extracting image of number plate for character segmentation
im = Image.open(IMAGE_NAME)
width, height = im.size

ymin = int((boxes[0][0][0]*height))
xmin = int((boxes[0][0][1]*width))
ymax = int((boxes[0][0][2]*height))
xmax = int((boxes[0][0][3]*width))

tempImage = np.array(image[ymin:ymax,xmin:xmax])
img_item = "extractedPlate.jpg"
cv2.imwrite(img_item, tempImage)
#cv2.imshow('Extracted plate', tempImage)

# All the results have been drawn on image. Now display the image.
cv2.imshow('Object detector', image)

#image preprocessing
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
'''
###
height, width, numChannels = tempImage.shape
imgHSV = np.zeros((height, width, 3), np.uint8)
imgHSV = cv2.cvtColor(cv2.UMat(im),cv2.COLOR_BGR2HSV)
imgHue,imgSaturation,imgValue = cv2.split(imgHSV)
imgGrayscale = imgValue
###
'''
#image1 = cv2.imread('extractedPlate.jpg')
imgGrayscale = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
cv2.imwrite('GrayScale.png',imgGrayscale)

height, width = imgGrayscale.shape
imgTopHat = np.zeros((height, width, 1), np.uint8)
imgBlackHat = np.zeros((height, width, 1), np.uint8)
structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)
imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
imgMaxContrastGrayscale = imgGrayscalePlusTopHatMinusBlackHat
cv2.imwrite('tophatblackhat.png',imgMaxContrastGrayscale)

height, width = imgGrayscale.shape
imgBlurred = np.zeros((height, width, 1), np.uint8)
imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0) 
#imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
#cv2.imwrite('temp.png', imgBlurred)
cv2.imshow('preprocess',imgBlurred)
gray = cv2.threshold(imgBlurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
gray = cv2.medianBlur(gray, 3)
cv2.imshow('gray',gray)
cv2.imwrite('temp.png', gray)
text = pytesseract.image_to_string(Image.open('temp.png'))
print("Plate number detected - ")
#print(text)        
import re
puretext = re.sub(r'\W+','', text)
print (puretext)
"""
#using pytesseract
                                                  
image1 = cv2.imread('extractedPlate.jpg')
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
gray = cv2.medianBlur(gray, 3)
cv2.imwrite('temp.png', gray)
cv2.imshow('gray',gray)
text = pytesseract.image_to_string(Image.open('temp.png'))
print("Plate number detected are - ")
print(text)

"""

#sql connection
import MySQLdb

try:
    # Open database connection
    db = MySQLdb.connect(host="localhost",user="root",passwd="1234",db="lpr" )

    # prepare a cursor object using cursor() method
    cur = db.cursor()

    # execute SQL query using execute() method.
    cur.execute("update table1 set flag = 1, time = now() where no_plate = %s",[puretext])

    db.commit()
except(MySQLdb.Error, MySQLdb.Warning) as e:
    print(e)


# Press any key to close the image
cv2.waitKey(0)
# Clean up
cv2.destroyAllWindows()
