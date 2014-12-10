__author__ = 'backing'

import numpy as np
import cv2
import os
import shutil
import number_cut as nc
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics

'''
    @describe used only once, for writing number img data to file
                next time we use numpy.load to get data
    @return none
'''
def write_img_data():
    data = np.array([])
    label = np.array([])
    filecnt = 0
    for sub_dir in range(0,10):
        dir = 'data/' + str(sub_dir)
        for filename in os.listdir(dir):
            img = cv2.imread(dir + '/'+ filename, 0)
            img.reshape(1,-1)
            data = np.append(data,img)
            label = np.append(label,sub_dir)
            filecnt += 1
    data = data.reshape(filecnt, -1)
    label = label.reshape(filecnt,)
    np.save('data.npy',data)
    np.save('label.npy',label)

def test_accuracy():
    data = np.load('data/data.npy')
    label = np.load('data/label.npy')
    data_train, data_test, label_train, label_test = train_test_split(data,label,test_size=0.9,random_state = 42) 
    print data_train.shape, data_test.shape, label_train.shape, label_test.shape
    clf = svm.SVC(kernel = 'linear', C = 100)
    #clf = KNeighborsClassifier()
    clf.fit(data_train,label_train)
    predict = clf.predict(data_test)
    print predict
    print label_test
    
    print metrics.accuracy_score(label_test,predict)

def load_classifier():
    data = np.load('data.npy')
    label = np.load('label.npy')
    #clf = KNeighborsClassifier()
    clf = svm.SVC(kernel = 'linear', C = 100)
    clf.fit(data,label)
    return clf

'''
    @param clf classifier
    @param img img to be classified
    @return class value
'''
def pred(clf, img):
    return clf.predict(img.reshape(1,-1))
    
''' 
    @param clf classifier
    @param img image to be reconize
    @return recognize number
'''
def recognize_number(clf, img):
    imgs = nc.get_single_numbers(img)
    res = 0
    for number in imgs:
        digit = pred(clf, number)
        res = res * 10 + digit[0]
    return res



if __name__ == '__main__':
#    clf = load_classifier()
#    img = cv2.imread('data/train/2.jpg',0)
#    print img.shape
 
#    print predict(clf,img)

    test_accuracy()
#    write_img_data()
