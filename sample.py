import cv2
import numpy as np 

print(cv2.__version__)

img  = cv2.imread("F:/sample/sample.jpg")

g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

b = cv2.GaussianBlur(g, (11, 11), 0)

e = cv2.Canny(b, 30, 150)

(cnts, h) = cv2.findContours(e.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count = 0

for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    a=cv2.contourArea(c)
    # print("Area:", a)
    if a > 10000:
        labels = img[y:y + h, x:x + w]
        mask = np.zeros(img.shape[:2], dtype = "uint8")
        ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(mask, (int(centerX), int(centerY)), int(radius),255, -1)
        mask = mask[y:y + h, x:x + w]
        cv2.imshow("Masked labels", cv2.bitwise_and(labels, labels, mask = mask))
        count = count+1
        cv2.imwrite("image_{}.jpg".format(count), labels)
        print("Total Labels are {}".format(count))
        cv2.waitKey(0)

