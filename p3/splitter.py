import cv2
import numpy as np
import urllib
#  vidcap = cv2.VideoCapture(0)
#  success,image = vidcap.read()
count = 0
success = True
url='http://192.168.0.16:8080/shot.jpg'
while success:
    imgResp = urllib.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    cv2.imshow('Capture', img)
    resized = cv2.resize(img, (140, 80), interpolation=cv2.INTER_AREA)
    gray = np.asarray(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY))
    #  frame = gray.reshape(-1, 80, 140, 1)
    cv2.imwrite("frame%d.jpg" % count, gray)     # save frame as JPEG file      
    print(count)
    count += 1
    ch = 0xFF & cv2.waitKey(30)
    if ch == 27:
        break

    continue
    cv2.imshow('Capture', image)
    image = cv2.resize(image, (140, 80), interpolation=cv2.INTER_AREA)
    image = np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    print('Read a new frame: ', success, count)
    count += 1
    success,image = vidcap.read()
    ch = 0xFF & cv2.waitKey(30)
    if ch == 27:
        break

cv2.destroyAllWindows()
