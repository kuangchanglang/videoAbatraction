import numpy as np
import math_calc as mc
import cv2
    
'''
    @param img
    @param point_s (x, y)
    @param point_e (x, y)
    @return return subimage in rectangle shape from img
'''
def get_subimage(img, point_s, point_e):
    x1 = min(point_s[1],point_e[1])
    x2 = max(point_s[1],point_e[1])
    y1 = min(point_s[0],point_e[0])
    y2 = max(point_s[0],point_e[0])
    return img[x1:x2,y1:y2]


''' 
    @describe judge if img1 and img2 are almost same image
    @param img1 
    @param img2
    @return True if same image, otherwise false
'''
def same_img(img1, img2):
    dist = mc.diff(img1,img2)
    if mc.average(dist) < 12:
        return True
    return False


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
# on mouse event


'''
    @describe judge if this frame has score board, we have set a label image and saved in target_img[0](global)
    @param frame 
    @param logo team logo that select by user
    @param rect region of label_img
    @return True if this frame has label_img, otherwise false
'''
def has_score_board(frame, logo, rect):
    if len(rect) == 0:
        return False
    # we set target_img[0] be label img, such as team logo
    # when we find team logo, we find score board
    label_img = get_subimage(frame, rect[0], rect[1])
    if same_img(logo, label_img):
        return True
    else:
        return False


