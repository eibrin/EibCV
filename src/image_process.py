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

def diffRange():
    img_01 = cv2.imread('../res/pose01.jpg')
    hsv = cv2.cvtColor(img_01, cv2.COLOR_BGR2HSV)
    red1 = np.array([0, 50, 50])
    red2 = np.array([15,255,255])
    red3 = np.array([165,50,50])
    red4 = np.array([180,255,255])
    blue1 = np.array([90,50,50])
    blue2 = np.array([120,255,255])
    green1 = np.array([45,50,50])
    green2 = np.array([75,255,255])
    yellow1 = np.array([20,50,50])
    yellow2 = np.array([35,255,255])

    mask_blue = cv2.inRange(hsv, blue1, blue2)
    mask_green = cv2.inRange(hsv, green1, green2)
    mask_red = cv2.inRange(hsv, red1, red2)
    mask_red2 = cv2.inRange(hsv, red3, red4)
    mask_yellow = cv2.inRange(hsv, yellow1, yellow2)

    res_blue = cv2.bitwise_and(img_01, img_01, mask=mask_blue)
    res_green = cv2.bitwise_and(img_01, img_01, mask=mask_green)
    res_red1 = cv2.bitwise_and(img_01, img_01, mask=mask_red)
    res_red2 = cv2.bitwise_and(img_01, img_01, mask=mask_red2)
    res_red = cv2.bitwise_or(res_red1, res_red2)
    res_yellow = cv2.bitwise_and(img_01, img_01, mask=mask_yellow)

    imgs = {'original': img_01, 'blue':res_blue, 'green':res_green, 'red':res_red, 'yellow':res_yellow}
    for i, (k,v) in enumerate(imgs.items()):
        plt.subplot(2,3,i+1)
        plt.title(k)
        plt.imshow(v[:,:,::-1])
        plt.xticks([]); plt.yticks([])
    plt.show()

def seamlessAdd():
    img = cv2.imread('../res/pose01.jpg')
    mark = cv2.imread('../res/rose_small.png')

    mask = np.full_like(mark, 255)
    height, width = img.shape[:2]
    center = (width//2, height//2)
    print('width :'+str(width), str(height), str(center))
    mixed = cv2.seamlessClone(mark, img, mask, center, cv2.MIXED_CLONE)
    cv2.imshow('mixed', mixed)

def normHistTest():
    img = cv2.imread('../res/viral01.jpeg', cv2.IMREAD_GRAYSCALE)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_equal = cv2.equalizeHist(img_norm)
    hist = cv2.calcHist([img_norm], [0], None, [256], [0,255])
    hist_equal = cv2.calcHist([img_equal], [0], None, [256], [0,255])
    cv2.imshow('original', img)
    cv2.imshow('hist', img_norm)
    cv2.imshow('equal', img_equal)
    plt.subplot(2,1,1)
    plt.plot(hist)
    plt.subplot(2,1,2)
    plt.plot(hist_equal)
    plt.show()

def masking(img, bp, name):
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cv2.filter2D(bp, -1, disc, bp)
    _, mask = cv2.threshold(bp, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(name, result)

def backProject_manual(img, hsv_img, hist_roi):
    hist_img = cv2.calcHist([hsv_img], [0], None, [0,180, 0, 256])
    hist_rate = hist_roi / (hist_img+1)
    h,s,v, = cv2.split(hsv_img)
    bp = hist_rate[h.ravel(), s.ravel()]
    bp = np.minimum(bp, 1)
    bp = bp.reshape(hsv_img.shape[:2])
    cv2.normalize(bp, bp, 0, 255, cv2.NORM_MINMAX)
    bp = bp.astype(np.uint8)
    masking(img, bp, 'result_manual')
def nukiTest():
    name = 'nuki'
    img = cv2.imread('../res/model03.jpg')
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    draw = img.copy()

    (x,y, w, h) = cv2.selectROI(name, img, False)

    if w>0 and h> 0:
        roi = draw[y:y+h, x:x+w]
        cv2.rectangle(draw, (x, y), (x+w, y+h), (0,0,255),2 )
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist_roi = cv2.calcHist([hsv_roi], [0,1], None, [180,256],  [0,180,0,256])
        bp = cv2.calcBackProject([hsv_img], [0,1], hist_roi, [0,180,0,256],1)
        masking(img, bp, 'result_cv')
        cv2.imshow('hist_roi', hist_roi)
        cv2.imshow('hsv_roi', hsv_roi)
        cv2.imshow('bp', bp)
    cv2.imshow(name, draw)

def motionTest():
    thresh = 25
    max_diff = 5

    a,b,c = None, None, None
    cap = cv2.VideoCapture(2)
    cap.set((cv2.CAP_PROP_FRAME_WIDTH), 480)
    cap.set((cv2.CAP_PROP_FRAME_HEIGHT), 320)
    if cap.isOpened():
        ret, a = cap.read()
        ret, b = cap.read()
        while ret:
            ret, c = cap.read()
            draw = c.copy()
            if not ret:
                break
            a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            diff1 = cv2.absdiff(a_gray, b_gray)
            diff2 = cv2.absdiff(b_gray, c_gray)
            ret, diff1_t = cv2.threshold(diff1, thresh, 255, cv2.THRESH_BINARY)
            ret, diff2_t = cv2.threshold(diff2, thresh, 255, cv2.THRESH_BINARY)
            diff = cv2.bitwise_and(diff1_t, diff2_t)
            k=cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN,k)
            diff_cnt = cv2.countNonZero(diff)
            if diff_cnt>max_diff:
                nzero = np.nonzero(diff)
                cv2.rectangle(draw,(min(nzero[1]), min(nzero[0])), (max(nzero[1]), max(nzero[0])), (0,255,0),2)
                cv2.putText(draw, "motion detection", (10,30), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,255))
            stacked = np.hstack((draw, cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)))
            cv2.imshow("motion sensor", c)#stacked)
            a=b
            b=c

def frame_generation():
    cap = cv2.VideoCapture(2)
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


if __name__ == "__main__":
    #cv2.imshow("img", img)
    #roi_test(100,100,300,200)
    #setROI()
    #binary_test()
    #check_diff()
    #diffRange()
    #seamlessAdd()
    #normHistTest()
    #nukiTest()
    motionTest()
    #frame_generation()
    key_process()
