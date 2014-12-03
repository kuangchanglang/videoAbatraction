import numpy as np
import cv2

'''
    @describe find arround first row, first column, last row, last column 
                to see which should be background
                then convert background to black, content should be white
                
                0 - black, 255 - white                
    @param img binary_img
    @return return converted img
'''
def convert_background(binary_img):
    height, width = binary_img.shape
    # total sum arround first row, first column, last row, last column
    total = sum(binary_img[0:,0]) + sum(binary_img[0:,width-1]) \
            + sum(binary_img[0,0:]) + sum(binary_img[height-1,0:])
    
    # each corner add twice, we need to eliminate them
    total = total - binary_img[0][0] - binary_img[0][width-1] \
                - binary_img[height-1,0] - binary_img[height-1][width-1]
    white_cnt = total / 255 
    black_cnt = (height*2+width*2-4) - white_cnt
#    print 'w,b:', white_cnt, black_cnt
    if white_cnt < black_cnt:
        binary_img = ~binary_img # turn to white background
    
    # eliminate head and bottom which are all background color
    top = 0
    while top<height and sum(binary_img[top]) == 255 * width:
        top += 1
    bottom = height-1
    while bottom>top and sum(binary_img[bottom]) == 255 * width:
        bottom -= 1
        
    binary_img = binary_img[top:bottom,0:]
    return binary_img 
    
'''
    @describe get seperate position which column are all background
    @param binary_img with only 0 and 255
    @param background
    @return [(start,end),(start,end)...]
'''
def get_seperate_pos(binary_img, background = 255):
    rows, cols = binary_img.shape
    res = []
    start, end = 0, 0
    for i in range(cols):
        if sum(binary_img[0:,i]) == background * rows:
            if i==0 or i-start==1:
                start = i
            else:
                end = i
                res.append((start,end))
                start = i #next start
    if i-start > 3: # last 
        res.append((start,i))
    return res

'''
    @param img
    @param pos regions in such form: [(start_point,end_point)...]
    @return cut img into serveral subimgs, column from start_point to end_point
'''
def seperate(img, pos):
    imgs = []
    for po in pos:
        cut_img = img[0:,po[0]:po[1]+1]
        cut_img = cv2.resize(cut_img,(24,24))
        imgs.append(cut_img)
    return imgs

'''
    @describe convert color img to binary img
    @param img
    @return binary image
'''
def convert_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    binary_img =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    retval, binary_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_img

'''
    @describe first detect background, then cut into numbers
    @param img color img
    @return return each single number size 13*26 in black-and-white
'''
def get_single_numbers(img):
    img = convert_binary(img)
    img = convert_background(img)
    pos = get_seperate_pos(img)
    imgs = seperate(img, pos)
    return imgs

if __name__ == '__main__':
    img = cv2.imread('train/31.jpg')
    img = convert_binary(img)
    img = convert_background(img)
    cv2.imshow('haha',img)
    cv2.waitKey(0)
    
