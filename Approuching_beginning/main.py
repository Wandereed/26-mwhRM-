import cv2
import numpy as np
station_bgr = cv2.imread(r"E:\26RM Visual\Approuching_beginning\1280X1280.PNG")
station_hsv = cv2.cvtColor(station_bgr, cv2.COLOR_BGR2HSV)
station_h, station_s, station_v = cv2.split(station_hsv)
maskw_s = cv2.inRange(station_s, np.array([0]), np.array([30]))
maskw_v = cv2.inRange(station_v, np.array([240]), np.array([255]))
mask= cv2.bitwise_and(maskw_v, maskw_s)
bed_out = cv2.bitwise_and(station_bgr, station_bgr, mask=mask)
contoursstation, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
result_img = station_bgr.copy()
for contours in contoursstation:
    if cv2.contourArea(contours) < 500:
        continue
    epsilon = 0.03 * cv2.arcLength(contours, True)
    approx = cv2.approxPolyDP(contours, epsilon, True)
    if approx.shape[0] == 6:
        cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contours)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
cv2.imshow('Original', station_bgr)
cv2.imshow("result", result_img)
cv2.imwrite("result1.png",  result_img)
cv2.waitKey(0)

