import cv2
import numpy as np
import logging
import os

os.makedirs("logs", exist_ok=True)

cap = cv2.VideoCapture(1)

logging.basicConfig(
    filename="logs/camera.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s"
)

logging.info("程序开始")

cameraMatrix = np.load("K.npy")
distCoeffs = np.load("dist.npy")

objectPoints = np.array([
    [-40, 30, 0],
    [40, 30, 0],
    [40, -30, 0],
    [-40, -30, 0]
], dtype=np.float32)


def order_points(pts):
    pts = pts.reshape(4, 2)

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上
    rect[3] = pts[np.argmax(diff)] # 左下

    return rect


axis_points = np.array([
    [0, 0, 0],
    [100, 0, 0],
    [0, 100, 0],
    [0, 0, 100]
], dtype=np.float32)


while True:

    ret, frame = cap.read()

    if not ret:
        logging.error("摄像头读取失败")
        break

    frame = cv2.undistort(frame, cameraMatrix, distCoeffs)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    contoursstation, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    found = False  # ⭐关键：是否找到有效目标

    for contours in contoursstation:

        area = cv2.contourArea(contours)
        if area < 3000:
            continue

        epsilon = 0.03 * cv2.arcLength(contours, True)
        approx = cv2.approxPolyDP(contours, epsilon, True)

        # ⭐关键修复1：必须保证四点
        if len(approx) != 4:
            continue

        logging.info("角点生成成功")

        ordered = order_points(approx)
        imagePoints = np.array(ordered, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            objectPoints,
            imagePoints,
            cameraMatrix,
            distCoeffs
        )

        if not success:
            continue

        logging.info("PnP成功")
        logging.info("rvec=%s tvec=%s", rvec.flatten(), tvec.flatten())

        projected, _ = cv2.projectPoints(
            axis_points,
            rvec,
            tvec,
            cameraMatrix,
            distCoeffs
        )

        projected = np.int32(projected)

        origin = tuple(projected[0].ravel())

        # X轴
        cv2.line(frame, origin,
                 tuple(projected[1].ravel()),
                 (0, 0, 255), 2)

        # Y轴
        cv2.line(frame, origin,
                 tuple(projected[2].ravel()),
                 (0, 255, 0), 2)

        # Z轴
        cv2.line(frame, origin,
                 tuple(projected[3].ravel()),
                 (255, 0, 0), 2)

        cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)

        found = True
        break  # ⭐关键：只取第一个有效目标

    # ⭐关键修复2：避免空 approx 崩溃
    if not found:
        logging.info("未检测到有效四边形")

    cv2.imshow("camera", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()

logging.info("程序结束")