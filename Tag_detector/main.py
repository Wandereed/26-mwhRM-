import cv2
import numpy as np
import logging
import os
os.makedirs("logs", exist_ok=True)
cap = cv2.VideoCapture(0)
logging.basicConfig(
    filename="logs/camera.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s"
)

logging.info("程序开始")

cameraMatrix=np.load("K.npy")
distCoeffs=np.load("dist.npy")
objectPoints = np.array([[-40, 30, 0],
                         [40, 30, 0],
                         [40, -30, 0],
                         [-40, -30, 0]], dtype=np.float32)
def order_points(pts):
    pts = pts.reshape(4, 2)

    rect = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    # x+y
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    # y-x
    diff = np.diff(
        pts,
        axis=1
    )

    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect
axis_points=np.array([[0, 0, 0],
                         [100, 0, 0],
                         [0, 100, 0],
                         [0, 0, 100]], dtype=np.float32)
def hsvmask(hsv):
    bed_h, bed_s, bed_v = cv2.split(hsv)
    maskb_h = cv2.inRange(bed_h, np.array([1]), np.array([5]))
    maskb_s = cv2.inRange(bed_s, np.array([0]), np.array([255]))
    maskb_v = cv2.inRange(bed_v, np.array([0]), np.array([255]))
    maskbhs = cv2.bitwise_and(maskb_h, maskb_s)
    maskbhsv = cv2.bitwise_and(maskbhs, maskb_v)
    return maskbhsv


while True:
    ret, frame = cap.read()
    ret1, frame1 = cap.read()
    if not ret:
        logging.warning("摄像头读取失败")
        continue
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.GaussianBlur(hsvmask(hsv),(3, 3),0)
    contoursstation, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    logging.info("x")
    for contours in contoursstation:
        epsilon = 0.03 * cv2.arcLength(contours, True)
        approx = cv2.approxPolyDP(contours, epsilon, True)
        area = cv2.contourArea(contours)
        if area < 1000:
            continue
        if len(approx) != 4:
            continue
        ordered = order_points(approx)
        logging.info("角点坐标生成")
        logging.info("角点排序完成")
        logging.info(ordered)
        imagePoints = ordered
        cv2.drawContours(frame1, [approx], -1, (0, 255, 0), 2)
        _, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
        logging.info("旋转和平移向量")
        logging.info("rvec=%s,tvec=%s", rvec, tvec)
        projected_axis_points, _ = cv2.projectPoints(axis_points, rvec, tvec, cameraMatrix, distCoeffs)
        projected_axis_points = np.int32(projected_axis_points)
        for i in range(1, len(projected_axis_points)):
            cv2.line(frame, tuple(projected_axis_points[0].ravel()), tuple(projected_axis_points[i].ravel()),
                     (0, 255, 0), 2)
    cv2.imshow("camera1", frame1)
    cv2.imshow("camera",frame)
    cv2.imshow("mask", mask)
    key=cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
