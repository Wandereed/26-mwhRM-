import cv2
import numpy as np
bed_bgr = cv2.imread(r"E:\26RM Visual\colour cut\images\bed.png")

bed_hsv = cv2.cvtColor(bed_bgr, cv2.COLOR_BGR2HSV)
bed_h, bed_s, bed_v = cv2.split(bed_hsv)
maskb_h = cv2.inRange(bed_h, np.array([10]), np.array([50]))
maskb_s = cv2.inRange(bed_s, np.array([50]), np.array([200]))
maskb_v = cv2.inRange(bed_v, np.array([50]), np.array([200]))
maskw_s = cv2.inRange(bed_s, np.array([0]), np.array([40]))
maskbhs = cv2.bitwise_and(maskb_h, maskb_s)
maskbhsv = cv2.bitwise_and(maskbhs, maskb_v)
mask= cv2.bitwise_or(maskbhsv, maskw_s)
bed_out = cv2.bitwise_and(bed_bgr, bed_bgr, mask=mask)
cv2.imshow("bed", bed_out)
cv2.imwrite("bed_out1.png", bed_out)
cv2.waitKey(0)