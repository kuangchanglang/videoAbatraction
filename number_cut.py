import numpy as np
import model
import image_operate as op
import cv2

'''
    @description find arround first row, first column, last row, last column 
                to see which should be background
                then convert background to white, content should be black
                
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
        
    binary_img = binary_img[top:bottom+1,0:]
    return binary_img 
    
'''
    @description get seperate position which column are all background
    @param binary_img with only 0 and 255
    @param background
    @return [(start,end),(start,end)...]
'''
def get_seperate_pos(binary_img, background = 255):
    rows, cols = binary_img.shape
    res = []
    start, end = 0, 0
    for i in range(cols):
        # may have 3 noise points in middle of the column, this limit to white background
        # if all column are background except 3 noise and first 6 are all background, this column should be background
         if sum(binary_img[0:,i]) == background * rows:
#        if sum(binary_img[0:,i]) >= background * (rows-3) \
#            and sum(binary_img[0:4,i]) == background * 4 :
            if i==0 or i-start==1:
                start = i
            else:
                end = i
                if end - start > 2:
                    res.append((start,end))
                start = i #next start
    if i-start > 3: # last 
        res.append((start,i))
    return res

'''
    @description cut img into serveral subimgs, column from start_point to end_point
    @param img
    @param pos regions in such form: [(start_point,end_point)...]
    @return images where each image contains only one number
'''
def seperate(img, pos):
    imgs = []
    for po in pos:
        cut_img = img[0:,po[0]:po[1]+1]
        cut_img = cv2.resize(cut_img,(24,24))
        imgs.append(cut_img)
    return imgs

'''
    @description convert color img to binary img
    @param img
    @return binary image
'''
def convert_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    binary_img =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    retval, binary_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_img

'''
    @description first detect background, then cut into numbers
    @param img color img
    @return return each single number size 13*26 in black-and-white, and position of each number
'''
def get_single_numbers(img):
    img = convert_binary(img)
    img = convert_background(img)
    pos = get_seperate_pos(img)
    imgs = seperate(img, pos)
    return imgs, pos


'''
    @description cut each text from a board image
    @param board image in color
    @return imgs, pos
            imgs are each text in 24*24 size
            pos are position of each text
'''
def get_text(board):
    board = op.get_binary(board)
    cv2.imshow('bin', board)
    height, width = board.shape
#    print height,width
    left,right = 0,0
    imgs = []
    pos = []
    row = 0

    # eliminate left few columns that are not backgound
    while row < width and sum(board[0:,row]) != 0 and sum(board[0:,row])!=255 * height:
        row += 1

    while row < width:
        #sum of this row
        background = 0
        pixel_sum = sum(board[0:,row])
        while pixel_sum == 0 or pixel_sum == 255 * height:
        # eliminate background
            background = pixel_sum
            row += 1
            if row >= width:
                break
            pixel_sum = sum(board[0:,row])
#            print 'x',pixel_sum, row
                
        left = row
        if row >= width:
            break
        pixel_sum = sum(board[0:,row])
        while row < width and pixel_sum != 0 and pixel_sum != 255 * height:
            row += 1
            if row >= width:
                break
            pixel_sum = sum(board[0:,row])
            
        right = row
#        print left,right
        # if this interval satisfy our need 
        if right - left < 3 or right - left > 40:
            continue
        single = board[0:,left:right+1]
        # convert background to black
        if background == 0:
            single = ~single

        # eliminate top and bottom backgoound
        top = 0
        while top < height and sum(single[top,0:])==255*(right-left+1):
            top +=1 
        bottom = height - 1
        while bottom > top and sum(single[bottom,0:])==255*(right-left+1):
            bottom -= 1
        if bottom - top < 10 or bottom - top < height / 2:
            continue

        pos.append((left,top))
        img = cv2.resize(single[top:bottom+1],(24,24))

        imgs.append(img)
    return imgs, pos

'''
    @test
'''
def test_get_text(img):
    imgs, pos = get_text(img)
    print pos
    clf = model.load_classifier()
    for i,img in enumerate(imgs):
        cv2.imshow(str(i),img)
        print model.pred(clf,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread('1.jpg')
    test_get_text(img)
    
