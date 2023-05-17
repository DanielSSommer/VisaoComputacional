import numpy as np
import cv2
from PIL import Image

def get_limits(colors):
    lower_limits = []
    upper_limits = []

    for color in colors:
        c = np.uint8([[color]]) 
        hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

        if color == [0, 0, 255]: 
            lowerLimit1 = np.array([0, 100, 100], dtype=np.uint8)
            upperLimit1 = np.array([10, 255, 255], dtype=np.uint8)
            lowerLimit2 = np.array([160, 100, 100], dtype=np.uint8)
            upperLimit2 = np.array([180, 255, 255], dtype=np.uint8)

            lower_limits.append(lowerLimit1)
            lower_limits.append(lowerLimit2)
            upper_limits.append(upperLimit1)
            upper_limits.append(upperLimit2)
        else:
            lowerLimit = hsvC[0][0][0] - 10, 100, 100
            upperLimit = hsvC[0][0][0] + 10, 255, 255

            lowerLimit = np.array(lowerLimit, dtype=np.uint8)
            upperLimit = np.array(upperLimit, dtype=np.uint8)

            lower_limits.append(lowerLimit)
            upper_limits.append(upperLimit)

    return lower_limits, upper_limits

colors_to_detect = [
    [0, 255, 255],  
    [255, 0, 0],  
    [0, 0, 255],  
    [0, 255, 0]  
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimits, upperLimits = get_limits(colors=colors_to_detect)

    for lowerLimit, upperLimit in zip(lowerLimits, upperLimits):
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
