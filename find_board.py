import cv2
import time
import numpy as np
import math_calc as mc

'''
'''
def find_same_cnt(output_dir, video_path, frames_cnt, interval):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(3))
    height = int(video.get(4))
    print 'width:',width,' height:',height
    result = np.array([[0 for x in range(width)] for y in range(height)])
    
    skip_to = 3000    
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip_to)
        
    i = 0
    max_cnt = 0
    while(video.isOpened() and i < frames_cnt):
        skip_to += interval
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip_to)
    
        reval, cur = video.retrieve() # retrive one frame
#        cv2.imwrite(output_dir+str(i)+'.jpg', cur)
        if i!=0:
            diff = mc.diff(cur,last)
            for h in range(height):
                for w in range(width):
                    if sum(diff[h][w]) < 30:
                        result[h][w] = result[h][w] + 1
                        if result[h][w] > max_cnt:
                            max_cnt = result[h][w]
        last = cur        
        i = i + 1
        print i      
    video.release()
    print max_cnt
    return result, max_cnt

def main():
    output_dir = '2\\'
    video_path = 'D:\\BaiduYunDownload\\2.rmvb'
    frames_cnt = 200 # capture only first 10000 frames
    interval = 20 # capture picture each 10 frame
    result, max_cnt = find_same_cnt(output_dir, video_path, frames_cnt, interval)
    print 'max_cnt', max_cnt
    
    read = [0x0,0xFF,0x0]
    img = cv2.imread('4.bmp', flags = 3)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if max_cnt - result[i][j] < 50:
                print i,j,result[i][j]
                img[i][j] = read
#    cv2.imshow('hehe', img)  
    cv2.imwrite(output_dir + 'out.jpg', img)
#    cv2.waitKey(0)
        
    
def test_grab():
    video = cv2.VideoCapture(video_path)
       
    i = 0
    while(video.isOpened() and i < frames_cnt):
        skip_to += interval
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip_to)
        
#        for j in range(interval):
#            video.grab() # skip interval frames             
        reval, cur = video.retrieve() # retrive one frame
        i = i + 1
    video.release()
    
    
        
if __name__ == '__main__':
    tick = time.time()
    main()
    print time.time() - tick
        
