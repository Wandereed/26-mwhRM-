import cv2
import numpy as np

video_path = r"E:\26RM Visual\Approuching_beginning_video\reset.mp4"
cap = cv2.VideoCapture(video_path)
while True:
    ret, station_bgr = cap.read()
    if not ret:
        print("error")
        break
    station_hsv = cv2.cvtColor(station_bgr, cv2.COLOR_BGR2HSV)
    lower_cyan = np.array([10, 100, 100])
    upper_cyan = np.array([13, 255, 255])
    mask = cv2.inRange(station_hsv, lower_cyan, upper_cyan)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = station_bgr.copy()
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imshow('Original', station_bgr)
    cv2.imshow('Color Mask', mask)
    cv2.imshow('Detected Humans', result_img)
    cv2.waitKey(10)
cap.release()
cv2.destroyAllWindows()
