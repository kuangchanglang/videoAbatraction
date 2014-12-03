import numpy as np
import cv2
import time 
import math_calc as mc
import number_cut as nc
import histgram as hist

img = ''
rects = []
win_name = 'frame'
sub_imgs = []

def play(filepath):
    global img
    global rects
    cv2.namedWindow(win_name)
    cv2.cv.SetMouseCallback(win_name, on_mouse, 0)
    cap = cv2.VideoCapture(filepath)

    skip_to = 1000
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip_to)
    key_val = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        
#        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        binary_frame =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        cv2.imshow(win_name, img)
        c = cv2.waitKey(key_val)
        if c == ord('c'):
            key_val = 1
        elif c == ord('q'):
            break
        elif c == ord(' '):
            skip_to += 1000
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip_to)
        else:
            skip_to += 500 # skip each five frames
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip_to)
        if key_val == 1: # start to cut images
            cut_rects(img, rects)
    cap.release()
    cv2.destroyAllWindows()

'''
    @describe write black-white img to outputpath from input img
    @param img
    @param outputpath
    @return none
'''
def write_binary(img, outputpath):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    binary_img =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    retval, binary_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(outputpath, binary_img)
    
'''
    @param img
    @param point_s
    @param point_e
    @return return subimage in rectangle shape from img
'''
def get_subimage(img, point_s, point_e):
    x1 = min(point_s[1],point_e[1])
    x2 = max(point_s[1],point_e[1])
    y1 = min(point_s[0],point_e[0])
    y2 = max(point_s[0],point_e[0])
    return img[x1:x2,y1:y2]

# on mouse event
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
            sub_imgs.pop()
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
            img2 = get_subimage(img, point_s, point_e)
#            img2 = cv2.resize(img2,(26,26))
#            cv2.imwrite('heihei.jpg',img2)
            rects.append((point_s,point_e))
            sub_imgs.append(img2)
            print 'add region', point_s, point_e
#            print 'rectangle add ', sub_imgs


def same_region(img1, img2):
    dist = mc.diff(img1,img2)
    if mc.average(dist) > 20:
        retur True
    return False

'''
    @param img 
    @param rects regions of the image
    @global sub_imgs sub imgs that cut by mouse event in advance
    @return none
# cut rectangles if it satisfy the condition that it is similar to given ones
'''
img_idx = 0
def cut_rects(img, rects):
    global img_idx
    for i, rect in enumerate(rects):
        img2 = get_subimage(img, rect[0], rect[1])
#        dist = mc.diff(sub_imgs[i], img2)
#        if mc.average(dist) > 20:
        if not same_region(sub_imgs[i],img2):
            break
        imgs = nc.get_single_numbers(img2)
        for number in imgs:
            cv2.imwrite('data/train/'+str(img_idx)+'.jpg', number)
            img_idx += 1


def draw_img(filepath):
    global img
    img = cv2.imread(filepath)
    cv2.imshow('draw',img)
    cv2.cv.SetMouseCallback('draw', on_mouse, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    play(filepath = 'D:\\BaiduYunDownload\\5.mp4')
#    img = cv2.imread('train/2.jpg')
#    write_binary(img, 'tmp.jpg')
#    draw_img('4.bmp')
