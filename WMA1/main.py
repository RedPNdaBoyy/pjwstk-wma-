import cv2
import numpy as np



def Morphological_Operations(mask):
    kernel = np.ones((16, 16), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closing

def find_ball_center(mask, img):
    M = cv2.moments(mask)
    if M["m00"] != 1:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    img_with_center = img.copy()
    cv2.circle(img_with_center, (cX, cY), 5, (0, 255, 0), -1)
    return img_with_center


cap = cv2.VideoCapture('movingball.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('zmodyfikowany_plik.mp4', cv2.VideoWriter_fourcc(*'mp4v'),cap.get(cv2.CAP_PROP_FPS), (width,height))



fr=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    imgRGB = frame
    # cv2.imshow('imgRGB', imgRGB)

    imgHSV = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV)
    # cv2.imshow('HSV',imgHSV)

    Down1 = np.array([0, 100, 100])
    Up1 = np.array([10, 255, 255])

    mask = cv2.inRange(imgHSV, Down1, Up1)

    closingMask = Morphological_Operations(mask)
    centerBall = find_ball_center(closingMask, imgRGB)
    cv2.imwrite('klatka_{}.png'.format(fr),mask)
    out.write(centerBall)
    fr+=1


cap.release()
out.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

