import numpy as np
import cv2
import time 
import histgram as hist
import math_calc as mc

def play(filepath):
    cap = cv2.VideoCapture(filepath)
    tick = time.time()
    start = 10000
    size = 300
    i = 0
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 20000)
    while(cap.isOpened()):
        ret, frame = cap.read()
        
    #    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #    if i > start:
    #        save_path = 'D:\\BaiduYunDownload\\2\\'+str(i)+'.jpg'
    #        print save_path
    #                
    #        cv2.imwrite(save_path, frame)
        i = i + 1
        if i % 100 == 0:
            print i
    cap.release()
    tick2 = time.time()
    print tick2-tick
    print i

    cv2.destroyAllWindows()

video = ""
def on_change(pos):
    global video
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
    
def play_with_bar(filepath):
    global video
    win_name = 'haha'
    video = cv2.VideoCapture(filepath)
    frames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    print frames
    
    pos = 0
    cv2.namedWindow(win_name)
    cv2.cv.CreateTrackbar('hehe', win_name, pos, int(frames), on_change)

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

'''
    @describe test video time using all read method 
    @param filepath
    @return none
'''
def read_video_time(filepath):
    cap = cv2.VideoCapture(filepath)
    tick = time.time()
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret or i > 10000:
            break
        i += 1
        
    cap.release()
    tick2 = time.time()
    print tick2-tick
    print i

    cv2.destroyAllWindows()
    
'''
    @describe test video time using skip method below
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
    @param filepath 
    @return none
'''
def read_video_time_skip(filepath):
    cap = cv2.VideoCapture(filepath)
    tick = time.time()
    pos = 0
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret or pos > 10000:
            break
        i += 1
        pos += 5
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
        
    cap.release()
    tick2 = time.time()
    print tick2-tick
    print i

'''
    @describe test video time using grab 4 frames and read 1 frame
    @param filepath
    @return none
'''
def read_video_time_grab(filepath):
    cap = cv2.VideoCapture(filepath)
    tick = time.time()
    i = 0
    while(cap.isOpened()):
        if i%5 != 0:
            ret = cap.grab()
        else:
            ret, frame = cap.read()
        if not ret or i > 10000:
            break
        i += 1
        
    cap.release()
    tick2 = time.time()
    print tick2-tick
    print i

    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
if __name__ == '__main__':
#    play_with_bar(filepath = 'D:\\BaiduYunDownload\\2.rmvb')
#    test_same_cnt()
    filepath = 'D:\\BaiduYunDownload\\5.mp4'
    read_video_time_grab(filepath =filepath) 
    read_video_time(filepath = filepath)
    read_video_time_skip(filepath = filepath)
