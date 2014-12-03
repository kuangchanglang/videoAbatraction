import numpy as np


'''
    @describe 
        img1 and img2 must be in the same shape
        size(shape) should be 3
    @param img1
    @param img2
    @return difference of img1 and img2, we don't use img1-img2 since 
            dtype of img1 and img2 are unit8, (1-2) will be 255
'''
def diff(img1, img2):
    res = np.zeros((img1.shape),dtype=np.uint8)
    for h in range(img1.shape[0]):
        for w in range(img1.shape[1]):
            for c in range(img1.shape[2]):
                res[h][w][c] = max(img1[h][w][c],img2[h][w][c]) - min(img1[h][w][c],img2[h][w][c])
    return res 

'''
    @describe
        return count of same pixel in img1 and img2
        img1 and img2 must be in same size
    @param img1
    @param img2
    @return cnt
'''
def same_cnt(img1,img2):
    cnt = 0
    for h in range(img1.shape[0]):
        for w in range(img1.shape[1]):
            res = 0
            for c in range(img1.shape[2]):
                res += max(img1[h][w][c],img2[h][w][c]) - min(img1[h][w][c],img2[h][w][c])
        if res < 10:
            cnt += 1
    print cnt
    return cnt

'''
    @param img
    @return sum of every pixel in img, then average
'''
def average(img):
    res = 0
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            res += sum(img[h][w])
    res /= img.size * img.shape[2]
    print 'sum: ',res
    return res
