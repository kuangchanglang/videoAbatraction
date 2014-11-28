import numpy as np
import cv2
import time 

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
    
if __name__ == '__main__':
    play_with_bar(filepath = 'D:\\BaiduYunDownload\\2.rmvb')