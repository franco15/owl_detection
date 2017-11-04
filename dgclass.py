#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example the histogram intersection algorithm is used in order
#to classify eight different superheroes. The histogram intersection is
#one of the simplest classifier and it uses histograms as
#comparison to identify the best match between an input image and a model.

#import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import numpy as np
from matplotlib import pyplot as plt
from deepgaze.color_classification import HistogramColorClassifier

PATHTO = 'data/train/images/'

#Defining the classifier
my_classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128],
                                        hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')

model_1 = cv2.imread(PATHTO + 'EK000003.jpg')
model_2 = cv2.imread(PATHTO + 'EK000176.jpg')
model_3 = cv2.imread(PATHTO + 'EK000348.jpg')
model_4 = cv2.imread(PATHTO + 'EK000681.jpg')
model_5 = cv2.imread(PATHTO + 'EK000483.jpg')
model_6 = cv2.imread(PATHTO + 'EK000489.jpg')
model_7 = cv2.imread(PATHTO + 'EK000498.jpg')
model_8 = cv2.imread(PATHTO + 'EK000503.jpg')

my_classifier.addModelHistogram(model_1)
my_classifier.addModelHistogram(model_2)
my_classifier.addModelHistogram(model_3)
my_classifier.addModelHistogram(model_4)
my_classifier.addModelHistogram(model_5)
my_classifier.addModelHistogram(model_6)
my_classifier.addModelHistogram(model_7)
my_classifier.addModelHistogram(model_8)

image = cv2.imread(PATHTO + 'EK000037.jpg') #Load the image
#Get a numpy array which contains the comparison values
#between the model and the input image
comparison_array = my_classifier.returnHistogramComparisonArray(image, method="intersection")
#Normalisation of the array
#comparison_array = cv2.normalize(my_classifier, my_classifier)
comparison_distribution = comparison_array / np.sum(comparison_array)

#Printing the arrays
print("Comparison Array:")
print(comparison_array)
print("Distribution Array: ")
print(comparison_distribution)

#Plotting a bar chart with the probability distribution
#If you are comparing more than 8 superheroes you have to
#change the total objects variable and add new labels in
total_objects = 8
label_objects = ('Owl', 'Owl', 'Owl', 'Cow', 'Owls', 'Owls', 'Owl', 'Owls')
font_size = 20
width = 0.5
plt.barh(np.arange(total_objects), comparison_distribution, width, color='r')
plt.yticks(np.arange(total_objects) + width/2.,label_objects , rotation=0, size=font_size)
plt.xlim(0.0, 1.0)
plt.ylim(-0.5, 8.0)
plt.xlabel('Probability', size=font_size)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
