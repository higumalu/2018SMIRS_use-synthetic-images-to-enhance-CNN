# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 20:39:38 2018

@author: HIGUMA_LU
"""
import cv2
import os
import numpy as np
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


img_width, img_height = 180, 120
dir_patch = './test-data'    
model_path_1 = './models/nongen/model.h5'
model_weights_path_1 = './models/nongen/weights.h5'
model_path_2 = './models/usegen/model.h5'
model_weights_path_2 = './models/usegen/weights.h5'
model_path_3 = './models/mix/model.h5'
model_weights_path_3 = './models/mix/weights.h5'


def predict(file):  

  x = load_img(file, grayscale=True, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  if result[0] > result[1]:
    #print("Predicted answer: abnormal")
    answer = 0
  else:
    #print("Predicted answer: normal")
    answer = 1

  return answer

def load_pred():
    # set data patch
    img_size = [img_width,img_height]
    
    data = []
    all_label=[]
    all_image = []
    pred = []
    
    cate_list = os.listdir(dir_patch)
    
    for index_cate in range(len(cate_list)):
        patch_img = os.listdir(dir_patch + '/' + cate_list[index_cate])
    
        # check  catelogy  &  data num
        print(cate_list[index_cate], len(patch_img))
    
    
        for index_img in range(len(patch_img)):
            img = []
            label = np.zeros((index_img,1),dtype=np.uint8)
            patch_img[index_img] = dir_patch + '/' + cate_list[index_cate] + '/' + patch_img[index_img]
    
            # imread by gray_img  (2 channels in numpy array
            pred.append(predict(patch_img[index_img]))
            #pred_2.append(predict(patch_img[index_img],model_path_2,model_weights_path_2))
            #pred_3.append(predict(patch_img[index_img],model_path_3,model_weights_path_3))
          
            all_label.append(index_cate)
    
            Y_test = np.array(all_label)

    return Y_test, pred

model = load_model(model_path_1)
model.load_weights(model_weights_path_1)
Y_test, pred_1=load_pred()
model = load_model(model_path_2)
model.load_weights(model_weights_path_2)
Y_test, pred_2=load_pred()
model = load_model(model_path_3)
model.load_weights(model_weights_path_3)
Y_test, pred_3=load_pred()

fpr_1,tpr_1,_ = roc_curve(Y_test[:], pred_1[:])
fpr_2,tpr_2,_ = roc_curve(Y_test[:], pred_2[:])
fpr_3,tpr_3,_ = roc_curve(Y_test[:], pred_3[:])



AUC_1 = auc(fpr_1, tpr_1)
AUC_2 = auc(fpr_2, tpr_2)
AUC_3 = auc(fpr_3, tpr_3)

#help(roc_curve)
plt.plot(fpr_1, tpr_1, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % AUC_1)

plt.plot(fpr_2, tpr_2, color='deeppink',
         lw=2, label='ROC curve (area = %0.2f)' % AUC_2)

plt.plot(fpr_3, tpr_3, color='cornflowerblue',
         lw=2, label='ROC curve (area = %0.2f)' % AUC_3)



plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',label='Radom guess')
plt.title('ROC_curve')  
plt.ylabel('True Positive Rate')  
plt.xlabel('False Positive Rate')  
plt.legend(loc="lower right")
plt.show()
