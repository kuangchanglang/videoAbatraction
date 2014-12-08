import numpy as np
import cv2
import time 
import math_calc as mc
import number_cut as nc
import number_detect as nd
import histgram as hist
import model 

img = ''
rects = []
win_name = 'nba'
target_imgs = []
video = ""
step = 5 # we check frames each n steps, default 5
def work(filepath):
    global video
    global step
    video = cv2.VideoCapture(filepath)
    # frame per second
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    step = fps / 4 # we check 4 frames each second
        
    ret = select_score_pos(video)
    # we don't need windows any more, and that make algorithm faster
    cv2.destroyAllWindows()
    if ret:
        get_score_info(video)
        cut_video(video)

    video.release()

def select_score_pos(video):
    global img
    global rects
    cv2.namedWindow(win_name)
    cv2.cv.SetMouseCallback(win_name, on_mouse, 0)
    frames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    cv2.cv.CreateTrackbar('tracker', win_name, 0, int(frames), on_change)
    while(video.isOpened()):
        ret, img = video.read()
        if not ret:
            break
        cv2.imshow(win_name, img) 

        c = cv2.waitKey(0)
        if c == ord('c'):
            cv2.cv.SetMouseCallback(win_name, on_mouse_do_nothing, 0)
            if len(target_imgs) != 4:
                return True
            else:
                return False
        elif c == ord('q'):
            return False 

def get_score_info(video):
    global step
    score_a, score_b = 0,0
    pos = 0
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
    clf = model.load_classifier()
    while(video.isOpened()):
        break
        ret, frame = video.read()
        if not ret:
            break
        if has_score_board(frame, target_imgs[0], rects[0]):
            # score a
            img = op.get_subimage(img, rects[1][0], rects[1][1])
            num = model.recognize_number(clf, img)
            # score b
            img = op.get_subimage(img, rects[2][0], rects[2][1])
            num = model.recognize_number(clf, img)
            # 24 second board
            img = op.get_subimage(img, rects[3][0], rects[3][1])
            num = model.recognize_number(clf, img)

        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)



def cut_video(video):
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    print 'cut'

'''
    @describe regnize score in one frame
    @param clf classifier used to recognize img number
    @param img frame that to be show 
    @param rects rectangles where score displayed
    @return none
'''
def recognize(clf, img, rects):
    red = [0,0,0xff]
    global img_idx
    if len(rects) == 0:
        return 
    # we set target_img[0] be label img, such as team logo
    # when we find team logo, we find score board
    label_img = nd.get_subimage(img, rects[0][0],rects[0][1])
    if not nd.same_region(target_imgs[0], label_img):
        return 

    for i, rect in enumerate(rects):
        if i==0:
            continue
        img2 = nd.get_subimage(img, rect[0], rect[1])
#        dist = mc.diff(sub_imgs[i], img2)
#        if mc.average(dist) > 20:
        imgs = nc.get_single_numbers(img2)


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
            img2 = nd.get_subimage(img, point_s, point_e)
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
    work(filepath = 'D:\\BaiduYunDownload\\5.mp4')
    #test_same_cnt()
