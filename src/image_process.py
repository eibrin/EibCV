import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('../res/supraspinatus_rupture.jpg')
isDragging = False
_x0,_y0, _w,_h = -1,-1,-1,-1
img_roi = img[_y0:_y0+_h, _x0:_x0+_w]

def roi_test(x1, y1, x2, y2):
    roi = img[y1:y2, x1:x2]
    print(roi.shape)
    cv2.rectangle(roi, (0, 0), (x2-x1-1,y2-y1-1), (0,255,100))  # 시작위치는 ROI 기준이고 최종점에서 -1씩 해줘야 함
    #cv2.line(roi, (0, 0), (x2-x1-1, y2-y1-1), (255,0,0))
    cv2.imshow("img", img)

def key_process():
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break;
        elif key ==ord('a') or key==27:
            cv2.destroyAllWindows()
            break;
        elif key == ord('a'):
            pass
            #print('a pressed')
        else: continue

def setROI():
    x,y,w,h = cv2.selectROI('img', img, False)  #opencv 에서 제공하는 roi 설정함수: 스페이스 엔터로 설정, c로 취소
    if w and h:
        roi = img[y:y+h, x:x+w]
        cv2.imshow('cropped', roi)
        cv2.moveWindow('cropped', 0,0)
        cv2.imwrite('../res/cropped.jpg', roi)

def binary_test():
    img_gray = cv2.imread('../res/supraspinatus_rupture.jpg', cv2.IMREAD_GRAYSCALE)
    _, t_130 = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)
    t, t_otsu = cv2.threshold(img_gray, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    print('otsu threshold: ', t)

    blk_size = 9
    cVal = 5
    t_adap = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk_size, cVal)
    print('otsu threshold: ', t)

    plt.subplot(2,2,1)
    plt.imshow(img_gray, cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(t_130, cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(t_otsu, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(t_adap, cmap='gray')

    plt.show()

def check_diff():
    img_01 = cv2.imread('../res/_normal01.jpg')
    img_02 = cv2.imread('../res/_viral01.jpg')
    img1_gray = cv2.cvtColor(img_01, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img_02, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(img1_gray, img2_gray)
    _, diff = cv2.threshold(diff, 1,255, cv2.THRESH_BINARY)
    diff_red = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    diff_red[:,:,2] =0
    spot = cv2.bitwise_xor(img_02, diff_red)
    cv2.imshow('img1', img_01)
    cv2.imshow('img2', img_02)
    cv2.imshow('diff', diff)
    cv2.imshow('spot', spot)


if __name__ == "__main__":
    #cv2.imshow("img", img)
    #roi_test(100,100,300,200)
    #setROI()
    #binary_test()
    check_diff()
    key_process()
