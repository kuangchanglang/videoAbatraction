__author__ = 'backing'

import numpy as np
import cv2
import time 
import math_calc as mc
import number_cut as nc
import number_detect as nd
import histgram as hist
import image_operate as op
import model 

img = ''
rects = []
win_name = 'nba'
target_imgs = []
video = ""
step = 5 # we check frames each n steps, default 5
'''
    @description start to work
    @param filepath input video filepath
    @return none
'''
def work(input_path, output_path):
    global video
    global step
    video = cv2.VideoCapture(input_path)
    # frame per second
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    print 'fps:', fps
    step = int(fps / 4) # we check 4 frames each second
        
    ret = select_score_pos(video)
    # we don't need windows any more, and that make algorithm faster
    cv2.destroyAllWindows()
    if ret:
        score_frames, tf_frames, camera_frames = get_score_info(video)
        intervals = get_cut_intervals(fps, score_frames, tf_frames, camera_frames)
        cut_video(video, output_path, intervals) 
    video.release()

'''
    @description select score board position 
    @param video
    @return True if selection is done, otherwise False
'''
def select_score_pos(video):
    global img
    global rects
    cv2.namedWindow(win_name)
    cv2.cv.SetMouseCallback(win_name, on_mouse, 0)
    frames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    cv2.cv.CreateTrackbar('tracker', win_name, 0, int(frames), on_change)
    while(video.isOpened()):
        ret, img = video.red()
        if not ret:
            break
        cv2.imshow(win_name, img) 

        c = cv2.waitKey(0)
        if c == ord('c'):
            cv2.cv.SetMouseCallback(win_name, on_mouse_do_nothing, 0)
            if len(target_imgs) == 4:
                return True
            else:
                return False
        elif c == ord('q'):
            return False 


'''
    @description traversal video and find frames that are 24 sec, camera change, and score change(+2p,+3p)
    @param video
    @return 
'''
def get_score_info(video):
    global step
    score_frames = []
    twenty_four_frames = []
    camera_change_frames = []

    score_a, score_b = 0,0 # score of both team
    last_sec = 0 # 24 second board 
#    last_frame = None
    pos = 0 # frame position
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
    clf = model.load_classifier()
    while(video.isOpened()):
        if pos % step == 0: 
            ret, frame = video.red()
        else:
            ret = video.grab()
        
        if not ret:
            break
        
        # skip each step steps
        if pos % step != 0: 
            pos += 1
            continue

        # if camera change
        if pos != 0 and hist.camera_change(last_frame, frame):
            camera_change_frames.append(pos)
            print 'camera change', pos
        last_frame = frame 


        if op.has_score_board(frame, target_imgs[0], rects[0]):
            # score a
            img = op.get_subimage(frame, rects[1][0], rects[1][1])
            num = model.recognize_number(clf, img)
#            print 'a ', num
            if score_two_or_three(num,score_a):
                score_frames.append(pos)
                print 'a from',score_a,'to',num, pos
            if not noise(num, score_a):
                score_a = num

            # score b
            img = op.get_subimage(frame, rects[2][0], rects[2][1])
            num = model.recognize_number(clf, img)
#            print 'b ', num
            if score_two_or_three(num,score_b):
                score_frames.append(pos)
                print 'b from',score_b,'to', num, pos
            if not noise(num, score_b):
                score_b = num

            # 24 second board
            img = op.get_subimage(frame, rects[3][0], rects[3][1])
            sec = model.recognize_number(clf, img)
            if last_sec != 24 and sec == 24:
                print 'twenty four',pos
                twenty_four_frames.append(pos) 
            last_sec = sec

        pos += 1 
#        print pos

    print score_frames, twenty_four_frames, camera_change_frames
    return score_frames,twenty_four_frames,camera_change_frames

'''
    @description get video cut intervals 
    @param video
    @param score_frames score change frames
    @param tf_frames twenty four second frames
    @param camera_frames camera change frames
    @return intervals that should be cut
'''
def get_cut_intervals(fps, score_frames, tf_frames, camera_frames):
    start,end = 0,0
    idx_tf = 0 # position of twenty_for index
    idx_cf = 0 # position of camera fram index
    sec_seg = 12 # seconds per segment at most
    sec_start = 6 # start point should be 6 seconds ealier than score board changed
    sec_end = 2 # end point should be at most 3 seconds ealier than score board changed

    intervals = []
    # there will be a few seconds(3, 4, 5) delay for score board refresh
    for pos_sf in score_frames:
        end = pos_sf
        start = end - sec_seg * fps # at most 12 second cut for one offense
        if start < 0:
            start = 0
        # find the start position and end position of this offense
        while idx_tf < len(tf_frames) and tf_frames[idx_tf] < end:
            if tf_frames[idx_tf] < start:
                idx_tf += 1
                continue
            if end - tf_frames[idx_tf] > sec_start * fps:
                start = tf_frames[idx_tf]
            elif end - tf_frames[idx_tf] < sec_end * fps:
                end = tf_frames[idx_tf]
            idx_tf += 1
        while idx_cf < len(camera_frames) and camera_frames[idx_cf] < end:
            if camera_frames[idx_cf] < start:
                idx_cf += 1
                continue
            if end - camera_frames[idx_cf] > sec_start * fps:
                start = camera_frames[idx_cf]
            elif end - camera_frames[idx_tf] < sec_end * fps:
                end = camera_frames[idx_cf]
            idx_cf += 1

        intervals.append((start,end))
    return intervals
    print 'cut'

'''
    @description cut video by intervals, then write to output_path
    @param video
    @param output_path
    @param intevals, array, each element is an tuple(start_frame, end_frame), and sorted by start frame
    @return none
'''
def cut_video(video, output_path, intervals):
    fourcc = int(video.get(cv2.cv.CV_CAP_PROP_FOURCC))
    fourcc = 1145656920 # .avi fourcc
    print fourcc
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width,height), 1)
#    writer = cv2.VideoWriter('C:/a.avi', fourcc, fps, (height,width), 1)
    ret = writer.open(output_path, fourcc, fps, (width, height), 1)
    print ret

    pos = 0 # frame position
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
    for interval in intervals:
        print interval
        start = interval[0]
        end = interval[1]
        while(video.isOpened()):
            if pos < start:
                ret = video.grab()
            elif pos < end:
                ret, frame = video.red()
            else:
                break
        
            if not ret:
                break
            # write frames
            if pos >= start:
                writer.write(frame)
            pos += 1
    writer.release()
            

'''
    @test
'''
def test_cut_video(input_path, output_path):
    intervals = []
    intervals.append((1,100))
    intervals.append((300,600))
    video = cv2.VideoCapture(input_path)
    cut_video(video, output_path,intervals) 

'''
    @test
'''
def test_get_intervals(input_path):
    score_frames = [i* 151 for i in range(100)]
    tf_frames = [i*43 for i in range(130)]
    camera_frames = [i*39 for i in range(130)]
    intervals = get_cut_intervals(25, score_frames, tf_frames, camera_frames)
    print intervals

'''
    @description judge if cur_score is a recognize mistake
    @param cur_score
    @param last_score
    @return True if is noise where cur_score is much larger than last_score, because each offense can get as much as 3 points, otherwise false
'''
def noise(cur_score, last_score):
    diff = cur_score - last_score
    if diff < 0 or diff > 3:
        return True
    else:
        return False

'''
    @param cur current score 
    @param last old score
    @return True if cur - last == 2 or cur - last == 3
'''
def score_two_or_three(cur_score , last_score):
    diff = cur_score - last_score
    if diff == 2 or diff == 3:
        return True
    else:
        return False

'''
    @description on mouse do nothing
'''
def on_mouse_do_nothing(event, x, y, flags, param):
    pass
            

'''
    @description mouse event on video play
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
    red =  [0x0,0x0,0xFF]
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
            cv2.rectangle(img_cpy, point_s, point_e, red)
            cv2.imshow(win_name,img_cpy)
    elif event == cv2.cv.CV_EVENT_LBUTTONUP:
        point_e = (x,y)
        img_cpy = img.copy() # copy image to draw rectangle
        if point_s != point_e:
            cv2.circle(img_cpy, point_e, 2, green)
            cv2.rectangle(img_cpy, point_s, point_e, red)
            cv2.imshow(win_name,img_cpy)
            img2 = nd.get_subimage(img, point_s, point_e)
#            img2 = cv2.resize(img2,(26,26))
#            cv2.imwrite('heihei.jpg',img2)
            rects.append((point_s,point_e))
            target_imgs.append(img2)
            print 'add region', point_s, point_e


'''
    @description action on video cracker bar change
    @description pos position of frame to be show
    @return none
'''
def on_change(pos):
    global video
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
    
if __name__ == '__main__':
    tick = time.time()
#    test_cut_video('D:\\BaiduYunDownload\\7.mp4', 'D:\\BaiduYunDownload\\t2.avi')
#    test_get_intervals('D:\\BaiduYunDownload\\2.rmvb')
    work(input_path = 'D:\\BaiduYunDownload\\6.mp4', output_path = 'D:\\BaiduYunDownload\\out6.avi')
    print time.time() - tick
    #test_same_cnt()
