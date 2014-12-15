__author__ = 'backing'

import cv2
import time
import numpy as np
import math_calc as mc
import image_operate as op
import model
import number_cut as nc

'''
    @desciption find same pixel count in each position
    @param output_dir output intermedia image to output dir
    @param video_path input video path
    @param frames_cnt total frames read
    @param interval get frames with interval, not every frame
    @return result, max_cnt
    
'''
def find_same_cnt(output_dir, video_path, frames_cnt, interval):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(3))
    width = int(video.get(4))
    print 'width:',width,' width:',width
    result = np.array([[0 for x in range(width)] for y in range(width)])
    
    skip_to = 3000    
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip_to)
        
    i = 0
    max_cnt = 0
    while(video.isOpened() and i < frames_cnt):
        skip_to += interval
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip_to)
    
        reval, cur = video.read() # retrive one frame
#        cv2.imwrite(output_dir+str(i)+'.jpg', cur)
        if i!=0:
            diff = mc.diff(cur,last)
            for h in range(width):
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

'''
    @description find horizontal lines in video
    @param video_path
    @return top two lines appeared
'''
def find_lines(video_path):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    frames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)

    red =  [0x0,0x0,0xFF]
    step = int(fps * 3) # step 3 seconds 
    total_frames = 500
    print width, width, step
    pos = 0
    last = ''
    
    # skip first ten minutes
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, fps * 60 * 10)
    line_appear = np.zeros(width,dtype=np.uint8)
    i = 0
    while video.isOpened and i < total_frames:
        if pos % step == 0:
            ret, cur = video.read()
            i += 1
            print i
        else:
            ret = video.grab()
        if not ret:
            break

        if pos % step != 0:
            pos += 1
            continue
 
#        binary = op.get_binary(cur)
#        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL  ,cv2.CHAIN_APPROX_SIMPLE)  
#        cv2.drawContours(cur, contours,-1,(0,0,255),1)  

        binary = op.get_binary(cur) 
        edges = cv2.Canny(binary,50,150,apertureSize = 3)

        lines = cv2.HoughLines(edges,1,np.pi/180,200)
#        print edges, lines
        if lines != None:
            for rho, theta in lines[0]:
                # only horizontal line reserved
                if np.sin(theta) != 1:
                    continue
                # intersect is rho 
#                print rho 
                line_appear[int(rho)] += 1 
                cv2.line(cur,(0,rho),(width,rho),red,2)
        cv2.imshow('cur', cur)
        pos += 1
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
#        print pos

    max_cnt = max(line_appear)

    
    valid_lines = []
    last_h = -1
    for h in range(width):
        if line_appear[h] * 3 > max_cnt:
            if last_h == -1:
                last_h = h
            elif h - last_h > 10 and h - last_h < 50:
                valid_lines.append((last_h,h))
            last_h = h
     
    return valid_lines 


'''
    @description we find two line in last step, we now find where score of both team present
    @param lines array like [(up,down),(up,down)...]
                each element contains two lines
    @return none
'''
def find_num_pos(video, lines):
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    if len(lines) == 0:
        print 'no lines'
        return

    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    clf = model.load_classifier()
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, fps * 60 * 15)
    while(video.isOpened):
        ret, cur = video.read()
        if not ret:
            break

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        for line in lines:
            imgs, pos = nc.get_text(cur[line[0]+1:line[1]-1])
#        cv2.imshow('binary', board)
#        print len(imgs)
            for i, img in enumerate(imgs):
                num, prob = model.pred_prob(clf, img)
                if prob > 0.2:
                    continue
                cv2.putText(cur, str(int(num)),(pos[i][0],line[0]), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,0xff), thickness = 2)

        cv2.imshow('cur', cur)
        

#        binary = op.get_binary(cur) 
#        edges = cv2.Canny(binary,50,150,apertureSize = 3)

#        lines = cv2.HoughLines(edges,1,np.pi/180,200)
#        print edges, lines
#        if lines != None:

    video.release()
    cv2.destroyAllWindows()
    
'''
    @desperated
    @description
    @return 
'''
def find_unchange_pos(video_path):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    frames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    red =  [0x0,0x0,0xFF]
    step = int(frames / 500) # read 500 frames in total
    print width, height, step
    pos = 0
    last = ''
    while(video.isOpened):
        if pos % step == 0:
            ret, cur = video.read()
            bin_cur = op.get_binary(cur) 
        else:
            ret = video.grab()
        if not ret:
            break
        if pos % step != 0:
            pos += 1
            continue

        cv2.imshow('cur', cur)
        if pos != 0: # first frame
            print 'begin calc'
#            diff = mc.diff(cur,last)
            for h in range(height):
                for w in range(width):
                    if bin_cur[h][w]==last[h][w]:
                        cur[h][w] = red
                    continue
                    ch, cw = h,w # copy
                    while ch < height and bin_cur[ch][w]==bin_cur[h][w]: #almost same color
                        ch += 1
                    if ch - h < 10:
                        continue
                    while cw < width and bin_cur[h][cw]==bin_cur[h][w]:
                        cw += 1
                    if cw - w < 30:
                        continue
                    cv2.rectangle(cur, (cw,h),(w,ch), red)
            cv2.imshow('last', last)
            cv2.imshow('cur', cur)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        last = bin_cur
        pos += 1
        print pos
    cv2.destroyAllWindows()
    video.release()

'''
    @test
'''
def main():
    output_dir = '2\\'
    video_path = 'D:\\BaiduYunDownload\\2.rmvb'
    frames_cnt = 200 # capture only first 10000 frames
    interval = 20 # capture picture each 10 frame
    result, max_cnt = find_same_cnt(output_dir, video_path, frames_cnt, interval)
    print 'max_cnt', max_cnt
    
    red = [0x0,0x0,0xFF]
    img = cv2.imread('4.bmp', flags = 3)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if max_cnt - result[i][j] < 50:
                print i,j,result[i][j]
                img[i][j] = red
#    cv2.imshow('hehe', img)  
    cv2.imwrite(output_dir + 'out.jpg', img)
#    cv2.waitKey(0)
        
if __name__ == '__main__':
    tick = time.time()
#    find_unchange_pos('D:\\BaiduYunDownload\\5.mp4')
#    lines = find_lines('D:\\BaiduYunDownload\\6.mp4')
#    print lines
#    print up,down
    video = cv2.VideoCapture('D:\\BaiduYunDownload\\2.rmvb')
#    lines = [(589,617),(617,639),(648,661)]
    lines = [(334,355)]
    find_num_pos(video,lines)

    print time.time() - tick
        
