# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:42:50 2018

@author: HIGUMA_LU
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 180, 120
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, grayscale=True, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  if result[0] > result[1]:
    print("Predicted answer: abnormal")
    answer = 'abnormal'
  else:
    print("Predicted answer: normal")
    answer = 'normal'

  return answer

tp = 0
tn = 0
fp = 0
fn = 0

for i, ret in enumerate(os.walk('./test-data/nor')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: normal")
    result = predict(ret[0] + '/' + filename)
    if result == "normal":
      tn += 1
    else:
      fp += 1

for i, ret in enumerate(os.walk('./test-data/abnor')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: abnormal")
    result = predict(ret[0] + '/' + filename)
    if result == "abnormal":
      tp += 1
    else:
      fn += 1

"""
Check metrics
"""
print("True Positive: ", tp)
print("True Negative: ", tn)
print("False Positive: ", fp)  
print("False Negative: ", fn)

precision = tp / (tp + fp)
ACC = (tp + tn) / (tp + tn + fp + fn)
TPR = tp / (tp + fn)
TNR = tn / (fp + tn)
FPR = fp / (fp + tn)

print("Precision: ", precision)
print("ACC_accuracy: ", ACC)
print("TPR_sensitivity: ", TPR)
print("TNR_specificity: ",TNR)
print("FPR_false alarm rate: ", FPR)

f_measure = (2 * TPR * precision) / (TPR + precision)
print("F-measure: ", f_measure)