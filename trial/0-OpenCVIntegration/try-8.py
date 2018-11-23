import cv2
import matplotlib as plt
import numpy as np

# Open Video
capture = cv2.VideoCapture(0)

# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

while (capture.isOpened()):
    # Capture Frames
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    # Frame
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(hsvFrame, lower, upper)

    # Filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # Combine the filtered frame
    # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skinFrame = cv2.bitwise_and(frame, frame, mask=skinMask)

    grayFrame = cv2.cvtColor(skinFrame, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(grayFrame, (1,1), 0)
    ret, threshFrame = cv2.threshold(blurFrame, 60, 255, cv2.THRESH_BINARY)

    # Show the frame
    cv2.imshow("images", threshFrame)
    # cv2.imshow("images", np.hstack([frame, skin]))

    k = cv2.waitKey(5) & 0xFF
    if k == ord('q') or k == 27:
        break

capture.release()
cv2.destroyAllWindows()
