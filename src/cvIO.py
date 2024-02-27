import datetime
from datetime import time

import cv2
import numpy as np


def load_file():
    img_file = "../res/pose01.jpg"
    img = cv2.imread(img_file)
    img_gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    save_file = "../res/pose01_edited.jpg"
    if img is not None:
        # cv2.imshow('img', img)
        # cv2.imshow('gray', img_gray)
        # cv2.imwrite(save_file, img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print("Can't load image")

def frame_generation():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while True:
            ret, img = cap.read()
            if ret:
                cv2.imshow('camera', img)
                if cv2.waitKey(1) != -1:
                    break
            else:
                print('no frame')
                break
        else:
            print('cant open camera')
        cap.release()

def drawTest():
    img = np.full((480, 640, 3), 255, dtype = np.uint8)
    cv2.line(img, (0,0), (100,100), (255,120,50), 20, cv2.LINE_AA)
    cv2.rectangle(img, (20,20), (100,120), (56,100,120), 2) #-1) 은 채우기
    cv2.circle(img, (150,200), 50, (25,55,75)) #,cv2.FILLED)
    cv2.imshow('img', img)
    cv2.waitKey()

def drawText():
    str_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.putText(img, str_date, (50,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (10,60,10))
    cv2.imshow('img', img)
    cv2.waitKey()

if __name__ == "__main__":
    #frame_generation()
    drawText()


