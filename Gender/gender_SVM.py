
# coding: utf-8

# In[10]:

import os
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import numpy as np
import PIL.Image as Image
from sklearn import svm
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[43]:

def read_data(path,id,size =None):
    print "hello read"
    #gender = id
    X,gender = [],[]
    size = 128 ,128
    for image in os.listdir(path):
        #print "hello folder :%s" %(image) 
        img = Image.open(os.path.join(path,image))
        img = img.convert("RGB")
        #print img
        img = img.resize(size, Image.ANTIALIAS)
        X.append(np.asarray(img, dtype=np.uint8).ravel())
        gender.append(id)
    return [X,gender]
    
size = 128 ,128
#read_data ("man", 1 , size)
[X,x] = read_data ("woman", -1)
[Y, y] = read_data ("man" , 1)
#print X
[R, r] = [X+Y, x+y]
R_train, R_test, r_train, r_test = train_test_split(R, r)

# Default svm
predictor = svm.LinearSVC()   

predictor.fit(R_train, r_train)
r_pred = predictor.predict(R_test)

target_names = ['woman', 'man']
print "SVM Accuracy:", accuracy_score(r_test, r_pred)
print "Classification report:\n", classification_report(r_test, r_pred, target_names=target_names) 
print predictor.coef_


# In[ ]:



