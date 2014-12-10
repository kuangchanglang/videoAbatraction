import numpy as np
import cv2
import time 
import math_calc as mc
import number_cut as nc
import image_operate as op 
import model 

img = ''
rects = [] # rectangles that select by user mouse action

win_name = 'nba'
target_imgs = [] #we will have four target imgs select by user
# first one is team logo, second and third one are score board of both team
# 4-th is 24 second board
video = ""

def play(filepath):
    global img
    global rects
    global video
    cv2.namedWindow(win_name)
    cv2.cv.SetMouseCallback(win_name, on_mouse, 0)

    clf = model.load_classifier()
    pos = 0 # frame position
    video = cv2.VideoCapture(filepath)
    frames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    cv2.cv.CreateTrackbar('tracker', win_name, pos, int(frames), on_change)
    key_val = 0
    while(video.isOpened()):
        ret, img = video.read()
        if not ret:
            break
        if key_val == 0:
            cv2.imshow(win_name, img) 

        pos = pos + 1
        c = cv2.waitKey(key_val)
        if c == ord('c'):
            key_val = 1
            cv2.cv.SetMouseCallback(win_name, on_mouse_do_nothing, 0)
        elif c == ord('q'):
            break

        if key_val==1:
            if op.has_score_board(img, target_imgs[0], rects[0]):
                show_recognize(clf, img, rects[1:])
            cv2.imshow(win_name, img) 
    video.release()
    cv2.destroyAllWindows()

'''
    @describe regnize both teams' score and 24 sec board in one frame, then show them on frame
    @param clf classifier used to recognize img number
    @param img frame that to be show 
    @param rects rectangle positions where score and 24 sec board displayed
    @return none
'''
def show_recognize(clf, img, rects):
    red = [0,0,0xff]
    if len(rects) == 0:
        return 

    for i,rect in enumerate(rects):
        img2 = op.get_subimage(img, rect[0], rect[1])
        cv2.imshow(str(i),img2)
        res = model.recognize_number(clf, img2)

        cv2.putText(img, str(int(res)), rect[0], cv2.FONT_HERSHEY_SIMPLEX,0.8, red, thickness = 2)
#            print res 

'''
    @describe on mouse do nothing
'''
def on_mouse_do_nothing(event, x, y, flags, param):
    pass
            

'''
    @describe mouse event on video play
    @param event
    @param x x-alis in pixel
    @param y y-alis in pixel
    @param flags
    @param param
'''
point_s = (0,0)
def on_mouse(event, x, y, flags, param):
    global point_s
    global img
    global rects
    green = [0x0,0xFF,0x0]
    read =  [0x0,0x0,0xFF]
    if event == cv2.cv.CV_EVENT_LBUTTONDOWN:
        img_cpy = img.copy() # copy image to draw rectangle
        point_s = (x,y)
        cv2.circle(img_cpy, point_s, 1, green)
        cv2.imshow(win_name,img_cpy)
    elif event == cv2.cv.CV_EVENT_RBUTTONDOWN:
        print 'cancel last region'
        if len(rects) != 0:
            rects.pop()
            target_imgs.pop()
    elif event == cv2.cv.CV_EVENT_MOUSEMOVE and (flags & cv2.cv.CV_EVENT_FLAG_LBUTTON):
        img_cpy = img.copy() # copy image to draw rectangle
        point_e = (x,y)
        if point_s != point_e:
            cv2.circle(img_cpy, point_e, 2, green) # draw point
            cv2.rectangle(img_cpy, point_s, point_e, read)
            cv2.imshow(win_name,img_cpy)
    elif event == cv2.cv.CV_EVENT_LBUTTONUP:
        point_e = (x,y)
        img_cpy = img.copy() # copy image to draw rectangle
        if point_s != point_e:
            cv2.circle(img_cpy, point_e, 2, green)
            cv2.rectangle(img_cpy, point_s, point_e, read)
            cv2.imshow(win_name,img_cpy)
            img2 = op.get_subimage(img, point_s, point_e)
#            img2 = cv2.resize(img2,(26,26))
#            cv2.imwrite('heihei.jpg',img2)
            rects.append((point_s,point_e))
            target_imgs.append(img2)
            print 'add region', point_s, point_e


'''
    @describe action on video cracker bar change
    @describe pos position of frame to be show
    @return none
'''
def on_change(pos):
    global video
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
    
if __name__ == '__main__':
    play(filepath = 'D:\\BaiduYunDownload\\2.rmvb')
    #test_same_cnt()
