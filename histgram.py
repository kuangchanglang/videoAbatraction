__author__ = 'backing'

import numpy as np
import cv2
import time 

'''
    @param img
    @return histgram of img in rgb space
    @warning still some problem in this function
    @deprecated
'''
def get_hist_rgb(img):
    hist_size = [256, 256, 256] # [hsize,ssize]
    r_range = [0, 256]
    g_range = [0, 256]
    b_range = [0, 256]
    
    histgram = cv2.calcHist([img], [0,1,2], None, hist_size, r_range+g_range+b_range)
#    histgram = cv2.normalize(histgram)
    return histgram
    
'''
    @param img
    @return histgram of img in hvs space
'''
def get_hist_hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
    hist_size = [180, 250] # [hsize,ssize]
    h_range = [0, 180]
    s_range = [0, 256]
    histgram = cv2.calcHist([hsv_img], [0,1], None, hist_size, h_range+s_range)
    histgram = cv2.normalize(histgram)
    return histgram

'''
    @param img1 
    @param img2
    @return histgram similarity of img1 and img2, using correlation metric
'''
def get_hist_similarity(img1, img2):
    hist1 = get_hist_hsv(img1)
    hist2 = get_hist_hsv(img2)
    return cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL)

'''
    @param last_frame
    @param cur_frame
    @return true if cur_frame is much different with last_frame
'''
def camera_change(last_frame, cur_frame):
    similarity = get_hist_similarity(last_frame, cur_frame)
    if similarity < 0.5:
        return True
    else:
        return False
    

'''
    @describe play video and stop while occuring camera change
    @param filepath
    @return none
'''
def play(filepath):
    video = cv2.VideoCapture(filepath)
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 60000)
    tick = time.time()
    i = 0
    while(video.isOpened()): 
        ret, cur_frame = video.read()
        cv2.imshow('frame', cur_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

        if i==0:
            last_frame = cur_frame
        elif i%3==0: # skip one frame
            similarity = get_hist_similarity(last_frame, cur_frame)
            print similarity
            last_frame = cur_frame
            if similarity < 0.5:
                cv2.waitKey(0)
        i = i + 1
            
    video.release()
    tick2 = time.time()
    print tick2-tick
    print i

    cv2.destroyAllWindows()

if __name__ == '__main__':
    play(filepath = "D:\\BaiduYunDownload\\3.mp4")
