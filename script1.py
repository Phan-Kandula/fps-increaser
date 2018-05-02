import os
import time
import cv2


def vid_to_frame(name, destination):
    vid = cv2.VideoCapture(name)
    num = 0
    while vid.isOpened():
        name = "frame_"
        ret, frame = vid.read()
        if ret:
            #            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            name = destination + name + str(num) + ".png"
            cv2.imwrite(name, frame)
            num = num + 1
        else:
            break
    vid.release()


if __name__ == '__main__':
    start_time = time.time()
    vid_to_frame('fs1', os.getcwd() + "/fs1_images/")
    print("--- %s seconds ---" % (time.time() - start_time))
