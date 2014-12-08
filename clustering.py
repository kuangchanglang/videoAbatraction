__author__ = 'backing'

import numpy as np
import cv2
import os
import shutil
from sklearn.cluster import KMeans

def cluster():
    data = np.array([])
    filecnt = 0
    dir = 'data/x11'
    for filename in os.listdir(dir):
        img = cv2.imread(dir + '/'+ filename, 0)
        img.reshape(1,-1)
        data = np.append(data,img)
        filecnt += 1
    kmeans = KMeans(init='k-means++',n_clusters = 4, n_init = 10)
    data = data.reshape(filecnt, -1)
    kmeans.fit(data)
    print kmeans.labels_

    i = 0
    for filename in os.listdir(dir):
        lable = kmeans.labels_[i]
        dest_dir = 'data/x'+str(lable)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        shutil.copy(dir+'/'+filename, dest_dir)
        i += 1

if __name__ == '__main__':
    cluster()
