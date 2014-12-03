import numpy as np
import cv2
import time 
import histgram as hist
import math_calc as mc

def play(filepath):
    video = cv2.VideoCapture(filepath)
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 20000)
    while(video.isOpened()):
        ret, frame = video.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

video = ""
def on_change(pos):
    global video
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
    
def play_with_bar(filepath):
    global video
    win_name = 'nba'
    video = cv2.VideoCapture(filepath)
    frames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    
    pos = 0
    cv2.namedWindow(win_name)
    cv2.cv.CreateTrackbar('tracker', win_name, pos, int(frames), on_change)

    while(video.isOpened()):
        ret, frame = video.read()
        cv2.imshow(win_name,frame)

        pos = pos + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    video.release()

def test_hist():
    img1 = cv2.imread('data/tmp/1.jpg')
    img2 = cv2.imread('data/tmp/1.jpg')
    diff = hist.get_hist_similarity(img1,img2)
    print diff

def test_same_cnt():
    img1 = cv2.imread('data/tmp/1.jpg')
    img2 = cv2.imread('data/tmp/3.jpg')
    cnt = mc.same_cnt(img1, img2)
    print cnt
    
if __name__ == '__main__':
    play_with_bar(filepath = 'D:\\BaiduYunDownload\\2.rmvb')
    #test_same_cnt()
